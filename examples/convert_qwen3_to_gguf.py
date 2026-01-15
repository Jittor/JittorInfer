#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import logging
import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Literal, Sequence, TypeVar, cast
from itertools import chain
from transformers import AutoConfig

import math
import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = logging.getLogger("qwen3-to-gguf")

###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


class ModelBase:
    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    dry_run: bool
    hparams: dict[str, Any]
    model_tensors: dict[str, Callable[[], Tensor]]
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path
    remote_hf_model_id: str | None

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    # Mistral format specifics (Default False for Qwen)
    is_mistral_format: bool = False
    disable_mistral_community_chat_template: bool = False
    sentence_transformers_dense_modules: bool = False

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False,
                 small_first_shard: bool = False, hparams: dict[str, Any] | None = None, remote_hf_model_id: str | None = None,
                 disable_mistral_community_chat_template: bool = False,
                 sentence_transformers_dense_modules: bool = False, merge_qkv: bool = False, merge_ffn: bool = False, 
                 max_layers: int | None = None):
        
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.dry_run = dry_run
        self.remote_hf_model_id = remote_hf_model_id
        self.sentence_transformers_dense_modules = sentence_transformers_dense_modules
        self.merge_qkv = merge_qkv
        self.merge_ffn = merge_ffn
        self.max_layers = max_layers
        self.hparams = ModelBase.load_hparams(self.dir_model, self.is_mistral_format) if hparams is None else hparams
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model

        # Apply heuristics to figure out typical tensor encoding
        if self.ftype == gguf.LlamaFileType.GUESSED:
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})")
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        self.dequant_model()

        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file,
                                           split_max_tensors=split_max_tensors, split_max_size=split_max_size, dry_run=dry_run, small_first_shard=small_first_shard)
        self.disable_mistral_community_chat_template = disable_mistral_community_chat_template

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}

        if remote_hf_model_id is not None:
            is_safetensors = True
            logger.info(f"Using remote model with HuggingFace id: {remote_hf_model_id}")
            remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_hf_model(remote_hf_model_id)
            for name, remote_tensor in remote_tensors.items():
                tensors[name] = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)
            return tensors

        prefix = "model"
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, prefix, ".safetensors")
        is_safetensors: bool = len(part_names) > 0
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")

        tensor_names_from_index: set[str] = set()

        index_name = "model.safetensors" if is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name

        if index_file.is_file():
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                tensor_names_from_index.update(weight_map.keys())
                part_dict: dict[str, None] = dict.fromkeys(weight_map.values(), None)
                part_names = sorted(part_dict.keys())
        else:
            weight_map = {}

        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            if is_safetensors:
                from safetensors import safe_open
                model_part = safe_open(str(self.dir_model / part_name), framework='pt', device='cpu')
                for name in model_part.keys():
                    if self.lazy:
                        data_gen = lambda name=name, model_part=model_part: model_part.get_tensor(name)
                    else:
                        data_gen = lambda name=name, model_part=model_part: model_part.get_tensor(name)
                    tensors[name] = data_gen
            else:
                model_part = torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True)
                for name in model_part.keys():
                    data_torch: Tensor = model_part[name]
                    if self.lazy:
                        data_gen = lambda data=data_torch: data
                    else:
                        data_gen = lambda data=data_torch: data
                    tensors[name] = data_gen

        if len(tensor_names_from_index) > 0:
            tensor_names_from_parts = set(tensors.keys())
            if len(tensor_names_from_parts.symmetric_difference(tensor_names_from_index)) > 0:
                missing = sorted(tensor_names_from_index.difference(tensor_names_from_parts))
                extra = sorted(tensor_names_from_parts.difference(tensor_names_from_index))
                missing_files = sorted(set(weight_map[n] for n in missing if n in weight_map))
                if len(extra) == 0 and len(missing_files) > 0:
                    raise ValueError(f"Missing or incomplete model files: {missing_files}\nMissing tensors: {missing}")
                else:
                    raise ValueError(f"Mismatch between weight map and model parts.\nMissing: {missing}\nExtra: {extra}")

        return tensors

    def dequant_model(self):
        # Basic dequantization implementation stub
        # Qwen models typically don't need complex dequant logic unless they are bitnet/gptq variants
        # Keeping empty for simplicity unless specifically needed
        pass

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, gen in self.model_tensors.items():
            # Skip layers beyond max_layers BEFORE loading the tensor
            if self.max_layers is not None:
                bid = None
                for part in name.split("."):
                    if part.isdecimal():
                        bid = int(part)
                        break
                if bid is not None and bid >= self.max_layers:
                    logger.debug(f"Skipping tensor load for layer {bid} (max_layers={self.max_layers}): {name}")
                    continue
            yield name, gen()

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}")
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        raise NotImplementedError("set_gguf_parameters() must be implemented in subclasses")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        return False

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype
            # Optimize dtype conversion based on output format
            if data_torch.dtype not in (torch.float16, torch.float32):
                # For fp16 output, convert bf16 directly to fp16 (faster than bf16->fp32->fp16)
                if self.ftype == gguf.LlamaFileType.MOSTLY_F16 and data_torch.dtype == torch.bfloat16:
                    data_torch = data_torch.to(torch.float16)
                else:
                    data_torch = data_torch.to(torch.float32)

            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in (self.modify_tensors(data_torch, name, bid)):
                data = data_torch.numpy()
                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(name, new_name, bid, n_dims)

                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if isinstance(data_qtype, bool) and data_qtype is False:
                     # Heuristics for quantization types
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    else:
                        # Fallbacks
                        data_qtype = gguf.GGMLQuantizationType.F16

                # Fast path: skip quantize() for simple dtype conversions
                if data_qtype == gguf.GGMLQuantizationType.F32:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32, copy=False)
                elif data_qtype == gguf.GGMLQuantizationType.F16:
                    if data.dtype != np.float16:
                        data = data.astype(np.float16, copy=False)
                else:
                    # Use gguf quantization for other types
                    try:
                        data = gguf.quants.quantize(data, data_qtype)
                    except gguf.QuantError as e:
                        logger.warning("%s, %s", e, "falling back to F16")
                        data_qtype = gguf.GGMLQuantizationType.F16
                        data = data.astype(np.float16, copy=False)

                shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):
        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()
        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()
        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)
        logger.info("Set model parameters")
        self.set_gguf_parameters()
        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write_vocab(self):
        raise NotImplementedError("write_vocab() must be implemented in subclasses")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)
        part_names.sort()
        return part_names

    @staticmethod
    def load_hparams(dir_model: Path, is_mistral_format: bool):
        try:
            config = AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()
        except Exception as e:
            logger.warning(f"Failed to load model config from {dir_model}: {e}")
            logger.warning("Trying to load config.json instead")
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        return config


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_arch = get_model_architecture(self.hparams, self.model_type)
        if "text_config" in self.hparams:
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.rope_parameters = self.hparams.get("rope_parameters", self.hparams.get("rope_scaling")) or {}

    def set_vocab(self):
        self._set_vocab_gpt2()

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)
        total_params = self.gguf_writer.get_total_parameter_count()[0]
        output_type: str = self.ftype.name.partition("_")[2]
        if self.fname_out.is_dir():
            fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, self.metadata.size_label, output_type, model_type="LoRA" if total_params < 0 else None)
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)
        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        # Use max_layers if specified, otherwise use the actual block_count
        effective_block_count = min(self.block_count, self.max_layers) if self.max_layers is not None else self.block_count
        self.gguf_writer.add_block_count(effective_block_count)
        if self.max_layers is not None:
            logger.info(f"gguf: limiting block_count from {self.block_count} to {effective_block_count} (max_layers={self.max_layers})")
        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
        if (n_embd := self.find_hparam(["hidden_size", "n_embd", "dim"], optional=True)) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
        if (n_ff := self.find_hparam(["intermediate_size", "n_inner", "hidden_dim"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
        if (n_head := self.find_hparam(["num_attention_heads", "n_head", "n_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count(n_head)
        if (n_head_kv := self.find_hparam(["num_key_value_heads", "n_kv_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
        
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        if (rope_theta := rope_params.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
        
        if (f_rms_eps := self.find_hparam(["rms_norm_eps", "norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
        
        self.gguf_writer.add_file_type(self.ftype)

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError('Splitting the vocabulary is not supported')
        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token
        
        seems_special = token_text in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")
        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        return seems_special

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if not added_tokens_decoder[i].normalized:
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)
        return tokens, toktypes, tokpre

    def get_vocab_base_pre(self, tokenizer) -> str:
        # Simplified hash checker, just returning generic or None if not found
        # Qwen usually uses qwen2/gpt2 pre-tokenization
        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()
        
        # Hash for Qwen2
        if chkhsh == "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c":
            return "qwen2"
        # Hash for Qwen1.5
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            return "qwen2"
            
        logger.warning(f"Unknown pre-tokenizer hash: {chkhsh}")
        return "gpt2" # Fallback

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_qwen(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
        vocab_size = hparams["vocab_size"]
        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            assert len(merged) == 2
            merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        added_vocab = tokenizer.special_tokens
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **added_vocab}.items()}

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.CONTROL)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=False)
        special_vocab.merges = merges
        if len(special_vocab.special_token_ids) == 0:
            special_vocab._set_special_token("bos", tokenizer.special_tokens["<|endoftext|>"])
            special_vocab._set_special_token("eos", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab._set_special_token("unk", tokenizer.special_tokens["<|endoftext|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        from sentencepiece import SentencePieceProcessor
        tokenizer_path = self.dir_model / 'tokenizer.model'
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")
        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))
        vocab_size = self.find_hparam(["vocab_size"], optional=True) or tokenizer.vocab_size()
        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            if token_id >= vocab_size:
                break
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)
            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE
            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype
            
        return tokens, scores, toktypes
    
    def _set_vocab_interns1(self):
        tokens: list[str] = []
        toktypes: list[int] = []
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        vocab = getattr(tokenizer, 'vocab', tokenizer.get_vocab())
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        tokpre = self.get_vocab_base_pre(tokenizer)
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if not added_tokens_decoder[i].normalized:
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("bos", 151643)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _try_set_pooling_type(self) -> None:
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"] == "sentence_transformers.models.Pooling":
                    pooling_path = mod["path"]
                    break
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling["pooling_mode_mean_tokens"]:
                pooling_type = gguf.PoolingType.MEAN
            elif pooling["pooling_mode_cls_token"]:
                pooling_type = gguf.PoolingType.CLS
            elif pooling["pooling_mode_lasttoken"]:
                pooling_type = gguf.PoolingType.LAST
            else:
                raise NotImplementedError("Only MEAN, CLS, and LAST pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)


# Necessary Helper for Qwen vocab operations
class QwenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN

    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts


class Qwen2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.hf_arch == "Qwen2Model":
            name = f"model.{name}"  # map to Qwen2ForCausalLM tensors
        if "language_model." in name:
            name = name.replace("language_model.", "") # for InternVL
        if name.startswith("mlp") or name.startswith("multi_modal_projector") \
                or name.startswith("vision_model") or name.startswith("audio_tower") \
                or name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector"):
            # skip vision and audio tensors
            return []
        yield from super().modify_tensors(data_torch, name, bid)


class Qwen3Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.QWEN3

    # extra logic for rerank models
    is_rerank: bool = False
    is_tied_embeddings: bool = False
    token_false_id: int | None = None
    token_true_id: int | None = None
    
    # tensors cache for merging
    qkv_cache: dict[int, dict[str, Tensor]]
    ffn_cache: dict[int, dict[str, Tensor]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_cache = {}
        self.ffn_cache = {}

        # track for intern-s1-mini
        hparams = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

        # a bit hacky, but currently the only way to detect if this is a rerank model
        # ref: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
        readme_path = self.dir_model / "README.md"
        readme_text = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf-8") as f:
                readme_text = f.read()
        if "# Qwen3-Reranker" in readme_text:
            self._find_rerank_config()

    def set_vocab(self):
        # deal with intern-s1-mini
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return

        super().set_vocab()

    def _find_rerank_config(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)

        self.is_rerank = True
        self.is_tied_embeddings = self.hparams.get("tie_word_embeddings", False)
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self.sep_token_id = tokenizer.convert_tokens_to_ids("|")

        assert self.token_false_id is not None and self.token_true_id is not None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        
        # Set head_dim if explicitly provided in config
        if (head_dim := self.find_hparam(["head_dim"], optional=True)) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)
            logger.info(f"Setting head_dim from config: {head_dim}")
        
        if self.is_rerank:
            self.gguf_writer.add_pooling_type(gguf.PoolingType.RANK)
            self.gguf_writer.add_classifier_output_labels(["yes", "no"])
            self.gguf_writer.add_chat_template([{
                "name": "rerank",
                "template": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                            "<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            }])

    def _get_cls_out_tensor(self, data_torch: Tensor) -> Tensor:
        # extract "yes" and "no" tokens from the output lm_head tensor
        false_row = data_torch[self.token_false_id]
        true_row = data_torch[self.token_true_id]
        return torch.stack([true_row, false_row], dim=0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "model.vision_" in name:
            # skip multimodal tensors
            return []

        if self.is_rerank:
            is_tied_head = self.is_tied_embeddings and "embed_tokens" in name
            is_real_head = not self.is_tied_embeddings and "lm_head" in name
            if is_tied_head or is_real_head:
                cls_out_head = (
                    gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.CLS_OUT] + ".weight",
                    self._get_cls_out_tensor(data_torch),
                )
                if is_tied_head:
                    embed = (self.map_tensor_name(name), data_torch)
                    return [cls_out_head, embed]
                if is_real_head:
                    return [cls_out_head]
        
        # Handle QKV merging
        if self.merge_qkv and bid is not None:
            if ".self_attn.q_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['q'] = data_torch
                self.qkv_cache[bid]['q_name'] = name
                return []  # Don't output yet
            elif ".self_attn.k_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['k'] = data_torch
                return []  # Don't output yet
            elif ".self_attn.v_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['v'] = data_torch
                # Now we have all three, merge them
                if 'q' in self.qkv_cache[bid] and 'k' in self.qkv_cache[bid]:
                    merged_qkv = torch.cat([self.qkv_cache[bid]['q'], 
                                           self.qkv_cache[bid]['k'], 
                                           self.qkv_cache[bid]['v']], dim=0)
                    # Get the mapped q_proj name and modify it to qkv
                    q_mapped_name = self.map_tensor_name(self.qkv_cache[bid]['q_name'])
                    # Replace attn_q with attn_qkv (e.g., blk.0.attn_q.weight -> blk.0.attn_qkv.weight)
                    qkv_mapped_name = q_mapped_name.replace("attn_q.", "attn_qkv.")
                    # Clear cache for this block
                    del self.qkv_cache[bid]
                    return [(qkv_mapped_name, merged_qkv)]
                return []  # Waiting for other tensors
        
        # Handle FFN gate/up merging
        if self.merge_ffn and bid is not None:
            if ".mlp.gate_proj.weight" in name:
                if bid not in self.ffn_cache:
                    self.ffn_cache[bid] = {}
                self.ffn_cache[bid]['gate'] = data_torch
                self.ffn_cache[bid]['gate_name'] = name
                return []  # Don't output yet
            elif ".mlp.up_proj.weight" in name:
                if bid not in self.ffn_cache:
                    self.ffn_cache[bid] = {}
                self.ffn_cache[bid]['up'] = data_torch
                # Now we have both, merge them
                if 'gate' in self.ffn_cache[bid]:
                    merged_gate_up = torch.cat([self.ffn_cache[bid]['gate'], 
                                               self.ffn_cache[bid]['up']], dim=0)
                    # Get the mapped gate_proj name and modify it to gate_up
                    gate_mapped_name = self.map_tensor_name(self.ffn_cache[bid]['gate_name'])
                    # Replace ffn_gate with ffn_gate_up (e.g., blk.0.ffn_gate.weight -> blk.0.ffn_gate_up.weight)
                    gate_up_mapped_name = gate_mapped_name.replace("ffn_gate.", "ffn_gate_up.")
                    # Clear cache for this block
                    del self.ffn_cache[bid]
                    return [(gate_up_mapped_name, merged_gate_up)]
                return []  # Waiting for gate tensor

        return super().modify_tensors(data_torch, name, bid)


###### HELPER CLASSES ######

# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
    }

    # only used when byteswapping data. Only correct size is needed
    _dtype_byteswap_map: dict[torch.dtype, type] = {
        torch.float64: np.float64,
        torch.float32: np.float32,
        torch.bfloat16: np.float16,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.uint64: np.uint64,
        torch.int32: np.int32,
        torch.uint32: np.uint32,
        torch.int16: np.int16,
        torch.uint16: np.uint16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.uint8,
        torch.float8_e4m3fn: np.uint8,
        torch.float8_e5m2: np.uint8,
    }

    # used for safetensors slices
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[...] if len(s.get_shape()) == 0 else s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_local_tensor(cls, t: gguf.utility.LocalTensor) -> Tensor:
        def load_tensor(tensor: gguf.utility.LocalTensor) -> Tensor:
            def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
                if sys.byteorder == 'big':
                    tensor = tensor.view(dtype).byteswap(inplace=False)
                return tensor
            dtype = cls._dtype_str_map[tensor.dtype]
            numpy_dtype = cls._dtype_byteswap_map[dtype]
            return torch.from_numpy(byteswap_tensor(tensor.mmap_bytes(), numpy_dtype)).view(dtype).reshape(tensor.shape)
        dtype = cls._dtype_str_map[t.dtype]
        shape = t.shape
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(t,), func=lambda r: load_tensor(r))
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
            if sys.byteorder == 'big':
                tensor = tensor.view(dtype).byteswap(inplace=False)
            return tensor
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        numpy_dtype = cls._dtype_byteswap_map[dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(meta=meta, args=(remote_tensor,), func=lambda r: torch.from_numpy(byteswap_tensor(np.frombuffer(r.data(), dtype=numpy_dtype), numpy_dtype)).view(dtype).reshape(shape))
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.numpy:
            return args[0].numpy()
        return cls._wrap_fn(func)(*args, **kwargs)

class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE
    
    # tensors cache for merging QKV
    qkv_cache: dict[int, dict[str, Tensor]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_cache = {}

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: expert used count = {n_experts_used}")
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        name = name.replace("language_model.", "") # InternVL
        
        # Handle QKV merging (only for attention, not for FFN as MoE FFN is special)
        if self.merge_qkv and bid is not None:
            if ".self_attn.q_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['q'] = data_torch
                self.qkv_cache[bid]['q_name'] = name
                return []  # Don't output yet
            elif ".self_attn.k_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['k'] = data_torch
                return []  # Don't output yet
            elif ".self_attn.v_proj.weight" in name:
                if bid not in self.qkv_cache:
                    self.qkv_cache[bid] = {}
                self.qkv_cache[bid]['v'] = data_torch
                # Now we have all three, merge them
                if 'q' in self.qkv_cache[bid] and 'k' in self.qkv_cache[bid]:
                    merged_qkv = torch.cat([self.qkv_cache[bid]['q'], 
                                           self.qkv_cache[bid]['k'], 
                                           self.qkv_cache[bid]['v']], dim=0)
                    # Get the mapped q_proj name and modify it to qkv
                    q_mapped_name = self.map_tensor_name(self.qkv_cache[bid]['q_name'])
                    # Replace attn_q with attn_qkv (e.g., blk.0.attn_q.weight -> blk.0.attn_qkv.weight)
                    qkv_mapped_name = q_mapped_name.replace("attn_q.", "attn_qkv.")
                    # Clear cache for this block
                    del self.qkv_cache[bid]
                    return [(qkv_mapped_name, merged_qkv)]
                return []  # Waiting for other tensors

        # handle aggregated expert tensors
        # GGUF stores dimensions reversed from PyTorch, so:
        # PyTorch (A,B,C) -> GGUF writes [C,B,A] -> GGML reads ne={C,B,A}
        # Input shapes from HF: (n_expert, n_ff_exp, n_embd) or (n_expert, n_embd, n_ff_exp)
        # Expected GGML ne: {n_embd, n_ff_exp, n_expert} for gate/up, {n_ff_exp, n_embd, n_expert} for down
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            # Input: (n_expert=128, n_ff_exp=768, n_embd=2048)
            # Want GGML ne: {n_ff_exp, n_embd, n_expert} = {768, 2048, 128}
            # Need PyTorch: (128, 2048, 768) [reversed of GGML]
            # So: permute(0, 2, 1): (128, 768, 2048) -> (128, 2048, 768)
            permuted = data_torch.permute(0, 2, 1).contiguous()
            return [(self.map_tensor_name(mapped), permuted)]

        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-1] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            split_dim = data_torch.shape[-1] // 2
            gate = data_torch[..., :split_dim].contiguous()
            up = data_torch[..., split_dim:].contiguous()
            # Input gate/up: (n_expert=128, n_embd=2048, n_ff_exp=768)
            # Want GGML ne: {n_embd, n_ff_exp, n_expert} = {2048, 768, 128}
            # Need PyTorch: (128, 768, 2048) [reversed of GGML]
            # So: permute(0, 2, 1): (128, 2048, 768) -> (128, 768, 2048)
            base_name = name.removesuffix(".weight")
            base = base_name.rsplit('.', 1)[0]
            mapped_gate = f"{base}.gate_proj.weight"
            mapped_up = f"{base}.up_proj.weight"
            perm_gate = gate.permute(0, 2, 1).contiguous()
            perm_up = up.permute(0, 2, 1).contiguous()
            return [
                (self.map_tensor_name(mapped_gate), perm_gate),
                (self.map_tensor_name(mapped_up), perm_up),
            ]

        if name.startswith("mlp") or name.startswith("vision_model") or name.startswith("model.vision_tower") or name.startswith("model.multi_modal_projector") or name.startswith("model.visual"):
            # skip visual tensors
            return []
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")

class Qwen3MoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = ModelBase.load_hparams(self.dir_model, False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        
        # Set head_dim if explicitly provided in config
        if (head_dim := self.find_hparam(["head_dim"], optional=True)) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)
            logger.info(f"Setting head_dim from config: {head_dim}")

    def set_vocab(self):
        # deal with intern-s1
        if self.origin_hf_arch == 'InternS1ForConditionalGeneration':
            self._set_vocab_interns1()
            return

        super().set_vocab()


###### MAIN LOGIC ######

def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")
    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")
    return n

def get_model_architecture(hparams: dict[str, Any], model_type: ModelType) -> str:
    text_config = hparams.get("text_config", {})
    vision_config = hparams.get("vision_config", {})
    arch = None
    if (arches := hparams.get("architectures")) is not None and len(arches) > 0:
        arch = arches[0]
    
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3 huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="f16",
        help="output format",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out a split plan and exit",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="(Experimental) Read safetensors file remotely",
    )
    parser.add_argument(
        "--merge-qkv", action="store_true",
        help="Merge Q, K, V projections into a single QKV weight tensor",
    )
    parser.add_argument(
        "--merge-ffn", action="store_true",
        help="Merge FFN gate and up projections into a single gate_up weight tensor",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Maximum number of layers to convert (for debugging, default: all layers)",
    )

    args = parser.parse_args()
    if args.model is None:
        parser.error("the following arguments are required: model")
    return args

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download
        allowed_patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"]
        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)

    if not dir_model.is_dir():
        logger.error(f'Error: {dir_model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        
        # Detect model architecture
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        arch = get_model_architecture(hparams, ModelType.TEXT)
        logger.info(f"Detected model architecture: {arch}")
        
        # Select model class based on architecture
        if arch in ["Qwen3MoeForCausalLM", "Qwen2MoeForCausalLM"]:
            model_class = Qwen3MoeModel
        else:
            model_class = Qwen3Model
        
        model_instance = model_class(dir_model, output_type, fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split,
                                     remote_hf_model_id=hf_repo_id,
                                     merge_qkv=args.merge_qkv, merge_ffn=args.merge_ffn, max_layers=args.max_layers)

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()