"""
Test script for:
  1. /extract-cnic-back  — CNIC back-side address extraction
  2. /verify-face        — Face verification (CNIC photo vs selfie)

Usage:
  python test_endpoints.py --back   <cnic_back_image>
  python test_endpoints.py --verify <cnic_front_image> <selfie_image>
  python test_endpoints.py --all    <cnic_back_image> <cnic_front_image> <selfie_image>
"""

import requests
import time
import json
import sys
import os
import argparse
from typing import Optional

API_URL = "https://cnic-processing-api-production.up.railway.app"
POLL_INTERVAL = 2    # seconds between status checks
MAX_WAIT      = 90   # seconds before giving up


# ─────────────────────────────── helpers ───────────────────────────────────

def check_health():
    print("=" * 55)
    print("  Health Check")
    print("=" * 55)
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        h = r.json()
        print(f"  API    : {h.get('status', '?')}")
        print(f"  Redis  : {h.get('redis', '?')}")
        print(f"  Service: {h.get('service', '?')}")
        print()
        return h.get("redis") == "connected"
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


def poll_result(task_id: str, label: str) -> Optional[dict]:
    """Poll /result/{task_id} until completed/failed or timeout."""
    print(f"\n⏳ Polling for result ({label}) — task: {task_id}")
    deadline = time.time() + MAX_WAIT

    for i in range(1, 999):
        time.sleep(POLL_INTERVAL)
        try:
            r = requests.get(f"{API_URL}/result/{task_id}", timeout=10)
        except Exception as e:
            print(f"  [{i}] Network error: {e}")
            continue

        if r.status_code == 404:
            print(f"  [{i}] Task not found yet...")
            if time.time() > deadline:
                break
            continue

        data = r.json()
        status = data.get("status", "unknown")
        print(f"  [{i}] Status: {status}")

        if status == "completed":
            return data
        elif status == "failed":
            print(f"\n  ❌ Task failed: {data.get('error')}")
            return data

        if time.time() > deadline:
            break

    print(f"\n  ❌ Timeout ({MAX_WAIT}s) waiting for task {task_id}")
    return None


def pretty(data: dict):
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ─────────────────────── 1. CNIC back extraction ───────────────────────────

def test_back_extraction(image_path: str):
    print("\n" + "=" * 55)
    print("  TEST: /extract-cnic-back")
    print("=" * 55)

    if not os.path.exists(image_path):
        print(f"  ❌ File not found: {image_path}")
        return False

    print(f"  Image : {image_path}  ({os.path.getsize(image_path):,} bytes)")

    with open(image_path, "rb") as f:
        files = {"cnic_back_image": (os.path.basename(image_path), f, "image/jpeg")}
        print(f"\n🚀 POST {API_URL}/extract-cnic-back")
        r = requests.post(f"{API_URL}/extract-cnic-back", files=files, timeout=30)

    print(f"   HTTP {r.status_code}")
    if r.status_code not in (200, 202):
        print(f"  ❌ Unexpected status: {r.text[:400]}")
        return False

    task = r.json()
    task_id = task.get("task_id")
    print(f"  task_id : {task_id}")
    print(f"  status  : {task.get('status')}")

    result = poll_result(task_id, "back-extraction")
    if not result:
        return False

    if result.get("status") == "completed":
        fields = result.get("result", {}).get("fields", {})
        print("\n✅ Extracted Address Fields:")
        print(f"  CNIC Number          : {fields.get('cnic_number')}")
        print(f"  موجودہ پتہ (Current) : {fields.get('mojooda_pata_urdu')}")
        print(f"  Current (Roman)      : {fields.get('mojooda_pata_roman')}")
        print(f"  مستقل پتہ (Permanent): {fields.get('mustaqil_pata_urdu')}")
        print(f"  Permanent (Roman)    : {fields.get('mustaqil_pata_roman')}")
        print(f"  Barcode Number       : {fields.get('barcode_number')}")
        print(f"  Confidence           : {fields.get('confidence')}")
        return True

    return False


# ─────────────────────── 2. Face verification ──────────────────────────────

def test_face_verify(cnic_path: str, selfie_path: str):
    print("\n" + "=" * 55)
    print("  TEST: /verify-face")
    print("=" * 55)

    for label, path in [("CNIC ", cnic_path), ("Selfie", selfie_path)]:
        if not os.path.exists(path):
            print(f"  ❌ File not found ({label}): {path}")
            return False
        print(f"  {label}: {path}  ({os.path.getsize(path):,} bytes)")

    with open(cnic_path, "rb") as cf, open(selfie_path, "rb") as sf:
        files = {
            "cnic_image"  : (os.path.basename(cnic_path),   cf, "image/jpeg"),
            "selfie_image": (os.path.basename(selfie_path), sf, "image/jpeg"),
        }
        print(f"\n🚀 POST {API_URL}/verify-face")
        r = requests.post(f"{API_URL}/verify-face", files=files, timeout=30)

    print(f"   HTTP {r.status_code}")
    if r.status_code not in (200, 202):
        print(f"  ❌ Unexpected status: {r.text[:400]}")
        return False

    task = r.json()
    task_id = task.get("task_id")
    print(f"  task_id : {task_id}")
    print(f"  status  : {task.get('status')}")

    result = poll_result(task_id, "face-verify")
    if not result:
        return False

    if result.get("status") == "completed":
        res = result.get("result", {})
        print("\n✅ Face Verification Result:")

        # Final verdict
        verified = res.get("final_verification")
        confidence = res.get("confidence")
        methods = res.get("methods_tried", [])

        verdict = "✅ MATCH" if verified else "❌ NO MATCH"
        print(f"  Verdict    : {verdict}")
        if confidence is not None:
            print(f"  Similarity : {confidence:.2f}%")
        print(f"  Methods    : {', '.join(methods) if methods else 'N/A'}")

        # Per-method details
        for method in ("face_recognition", "deepface", "opencv_histogram"):
            if method in res:
                m = res[method]
                print(f"\n  [{method}]")
                print(f"    is_match   : {m.get('is_match')}")
                print(f"    similarity : {m.get('similarity')}")

        return verified is not None

    return False


# ──────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CNIC API endpoint tester")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--back",   nargs=1,   metavar="CNIC_BACK",
                       help="Test back-extraction only")
    group.add_argument("--verify", nargs=2,   metavar=("CNIC_FRONT", "SELFIE"),
                       help="Test face-verify only")
    group.add_argument("--all",    nargs=3,
                       metavar=("CNIC_BACK", "CNIC_FRONT", "SELFIE"),
                       help="Run both tests")
    args = parser.parse_args()

    if not check_health():
        print("❌ API unavailable — aborting.")
        sys.exit(1)

    results = {}

    if args.back:
        results["back_extraction"] = test_back_extraction(args.back[0])

    elif args.verify:
        results["face_verify"] = test_face_verify(args.verify[0], args.verify[1])

    elif args.all:
        results["back_extraction"] = test_back_extraction(args.all[0])
        results["face_verify"]     = test_face_verify(args.all[1], args.all[2])

    # Summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for test, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {test}")
    print()


if __name__ == "__main__":
    main()