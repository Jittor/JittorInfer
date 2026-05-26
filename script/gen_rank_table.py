#!/usr/bin/env python3
"""Generate HCCN rank table JSON from ASCEND_RT_VISIBLE_DEVICES and /etc/hccn.conf.

Reads ASCEND_RT_VISIBLE_DEVICES (or ASCEND_VISIBLE_DEVICES) from the
environment, looks up each device's HCCN IP from /etc/hccn.conf, and writes a
rank_table.json to stdout using the current hostname as server_id.
"""

import json
import os
import socket
import sys


def parse_hccn_conf(path="/etc/hccn.conf"):
    """Parse /etc/hccn.conf, return dict mapping device_id (str) -> ip (str)."""
    addrs = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("address_"):
                    key, value = line.split("=", 1)
                    dev_id = key[len("address_"):]
                    addrs[dev_id] = value
    except FileNotFoundError:
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    return addrs


def main():
    devices_str = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if not devices_str:
        print("ERROR: ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES must be set", file=sys.stderr)
        sys.exit(1)

    device_ids = [d.strip() for d in devices_str.split(",") if d.strip()]
    if not device_ids:
        print("ERROR: no visible devices found", file=sys.stderr)
        sys.exit(1)

    hccn_addrs = parse_hccn_conf()

    server_id = socket.gethostname()

    devices = []
    for rank_id, dev_id in enumerate(device_ids):
        ip = hccn_addrs.get(dev_id, "unknown")
        devices.append({
            "device_ip": ip,
            "device_id": dev_id,
            "rank_id": str(rank_id),
        })

    rank_table = {
        "server_count": "1",
        "version": "1.0",
        "status": "completed",
        "server_list": [
            {
                "device": devices,
                "server_id": server_id,
                "host_nic_ip": "reserve",
            }
        ],
    }

    json.dump(rank_table, sys.stdout, indent=4)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
