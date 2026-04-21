

import re

import tldextract
from urllib.parse import urlparse, parse_qs, unquote
import ipaddress
import math




def is_ip(domain):
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False

def extract_structural_features(url):
    parsed = urlparse(url)

    return {
        "domain": parsed.netloc,
        "subdomain": parsed.hostname,
        "tld": parsed.netloc.split('.')[-1] if parsed.netloc else "",
        "path": parsed.path,
        "query_params": list(parse_qs(parsed.query).keys()),
        "fragment": parsed.fragment
    }

def extract_domain_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    full_domain = parsed.hostname or ""
    registered_domain = ".".join(part for part in [ext.domain, ext.suffix] if part)

    subdomains = ext.subdomain.split('.') if ext.subdomain else []

    return {
        "registered_domain": registered_domain,
        "full_domain": full_domain,
        "is_ip_address": is_ip(full_domain),
        "domain_length": len(full_domain),
        "num_subdomains": len(subdomains)
    }



def detect_homograph(domain):
    # Simple heuristic: presence of non-ascii characters
    try:
        domain.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True

def extract_spoofing_features(url):
    parsed = urlparse(url)
    domain = parsed.hostname or ""

    return {
        "homograph_detected": detect_homograph(domain)
    }

def shannon_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob])

def has_base64_pattern(s):
    # crude heuristic: long base64-like string
    return bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', s))

def extract_encoding_features(url):
    decoded_url = unquote(url)

    return {
        "contains_url_encoding": "%" in url,
        "num_encoded_chars": url.count('%'),
        "contains_base64": has_base64_pattern(url),
        "contains_hex_encoding": bool(re.search(r'0x[0-9a-fA-F]+', url)),
        "url_entropy": round(shannon_entropy(url), 3),
        "decoded_differs": decoded_url != url
    }

def extract_protocol_features(url):
    parsed = urlparse(url)

    return {
        "protocol": parsed.scheme,
        "uses_https": parsed.scheme == "https",
        "port": parsed.port if parsed.port else (443 if parsed.scheme == "https" else 80),
        "has_explicit_port": parsed.port is not None
    }

def extract_all_features(url):
    features = {}

    features["url"] = url
    features["structural"] = extract_structural_features(url)
    features["domain"] = extract_domain_features(url)
    features["spoofing"] = extract_spoofing_features(url)    
    features['encoding'] = extract_encoding_features(url)
    features['protocol'] = extract_protocol_features(url)

    return features

def deriveUrlFeatures(url_before,url_after):
    features_before = extract_all_features(url_before)
    features_after = extract_all_features(url_after)

    # You can also add derived features comparing before and after
    derived_features = {
        "domain_changed": features_before["domain"]["registered_domain"] != features_after["domain"]["registered_domain"],
        "protocol_changed": features_before["protocol"]["protocol"] != features_after["protocol"]["protocol"],
        "encoding_changed": features_before["encoding"]["contains_url_encoding"] != features_after["encoding"]["contains_url_encoding"],        
    }

    return {
        "before": features_before,
        "after": features_after,
        "derived": derived_features,
        # "is_malicious_after":  maliciousurl_tool.is_malicious(url_after)
    }
