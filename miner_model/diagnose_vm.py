#!/usr/bin/env python3
"""
Diagnostic script to check VM connectivity for Bittensor miner.
Run this on your Ubuntu VM to identify network/firewall issues.
"""

import socket
import sys
import subprocess
import requests
from urllib.parse import urlparse

def test_port(host, port, timeout=5):
    """Test if a port is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"  ❌ Error testing {host}:{port}: {e}")
        return False

def test_http_endpoint(url, timeout=10):
    """Test if an HTTP/HTTPS endpoint is reachable."""
    try:
        response = requests.get(url, timeout=timeout, verify=True)
        return response.status_code == 200
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_websocket_endpoint(url, timeout=10):
    """Test if a WebSocket endpoint is reachable."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'wss' else 80)
        return test_port(host, port, timeout)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def get_external_ip():
    """Get the VM's external IP address."""
    try:
        response = requests.get('https://api.ipify.org', timeout=5)
        return response.text.strip()
    except:
        return "Unable to determine"

def check_firewall():
    """Check if UFW firewall is active."""
    try:
        result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
        return 'Status: active' in result.stdout
    except:
        return False

def main():
    print("=" * 70)
    print("Bittensor Miner VM Diagnostic Tool")
    print("=" * 70)
    print()
    
    # 1. Check external IP
    print("1. Checking external IP address...")
    external_ip = get_external_ip()
    print(f"   External IP: {external_ip}")
    print()
    
    # 2. Check firewall status
    print("2. Checking firewall status...")
    firewall_active = check_firewall()
    if firewall_active:
        print("   ⚠️  UFW firewall is ACTIVE")
        print("   ⚠️  You may need to allow ports for Bittensor:")
        print("      sudo ufw allow 8091/tcp")
        print("      sudo ufw allow 30333/tcp")
    else:
        print("   ✅ UFW firewall is inactive or not installed")
    print()
    
    # 3. Test blockchain endpoints
    print("3. Testing blockchain connectivity...")
    
    # Testnet endpoints
    testnet_endpoints = [
        ("wss://entrypoint-finney.opentensor.ai:443", "Testnet WebSocket"),
        ("https://entrypoint-finney.opentensor.ai:443", "Testnet HTTPS"),
    ]
    
    for endpoint, name in testnet_endpoints:
        print(f"   Testing {name}: {endpoint}")
        if endpoint.startswith('wss://'):
            reachable = test_websocket_endpoint(endpoint)
        else:
            reachable = test_http_endpoint(endpoint)
        
        if reachable:
            print(f"   ✅ {name} is reachable")
        else:
            print(f"   ❌ {name} is NOT reachable")
            print(f"      This will prevent the miner from syncing with the blockchain!")
    print()
    
    # 4. Test if axon port is open (from external)
    print("4. Testing axon port accessibility...")
    print("   Note: This tests if the port is open locally.")
    print("   For full test, try connecting from an external machine:")
    print(f"   telnet {external_ip} 8091")
    
    # Test if port is listening locally
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 8091))
        sock.close()
        if result == 0:
            print("   ✅ Port 8091 is open locally")
        else:
            print("   ⚠️  Port 8091 is not listening (this is OK if miner isn't running)")
    except:
        print("   ⚠️  Could not test port 8091")
    print()
    
    # 5. Check GCP firewall rules
    print("5. GCP Firewall Configuration:")
    print("   ⚠️  IMPORTANT: Check your GCP firewall rules!")
    print("   The following ports need to be open:")
    print("   - Port 8091 (or your chosen axon port): TCP, Inbound")
    print("   - Port 30333: TCP, Inbound/Outbound (for blockchain sync)")
    print()
    print("   To check GCP firewall rules:")
    print("   gcloud compute firewall-rules list")
    print()
    print("   To create a firewall rule:")
    print("   gcloud compute firewall-rules create allow-bittensor-miner \\")
    print("     --allow tcp:8091 \\")
    print("     --source-ranges 0.0.0.0/0 \\")
    print("     --description 'Allow Bittensor miner axon port'")
    print()
    
    # 6. DNS resolution test
    print("6. Testing DNS resolution...")
    test_hosts = [
        "entrypoint-finney.opentensor.ai",
        "api.ipify.org",
    ]
    for host in test_hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"   ✅ {host} resolves to {ip}")
        except Exception as e:
            print(f"   ❌ {host} DNS resolution failed: {e}")
    print()
    
    print("=" * 70)
    print("Diagnostic complete!")
    print("=" * 70)
    print()
    print("Common fixes:")
    print("1. If blockchain endpoints are unreachable:")
    print("   - Check GCP firewall rules for outbound traffic")
    print("   - Check if your VM has internet access: ping 8.8.8.8")
    print()
    print("2. If axon port is not accessible:")
    print("   - Create GCP firewall rule to allow TCP port 8091")
    print("   - Ensure VM has external IP or is behind a load balancer")
    print()
    print("3. If miner hangs during sync:")
    print("   - Add --logging.debug flag for more verbose output")
    print("   - Check if you're registered: btcli wallet overview --netuid 420 --subtensor.network test")

if __name__ == "__main__":
    main()
