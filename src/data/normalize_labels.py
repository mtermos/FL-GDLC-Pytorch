import pandas as pd

# 1. Define your unified label mapping
LABEL_MAPPING = {
    # benign
    "BENIGN":          "benign",
    "Benign":          "benign",

    # scanning / portscan
    "PortScan":        "scanning",
    "scanning":        "scanning",

    # dos variants
    "DoS Hulk":        "dos",
    "DoS slowloris":   "dos",
    "DoS Slowhttptest": "dos",
    "DoS GoldenEye":   "dos",
    "dos":             "dos",

    # ddos
    "DDoS":            "ddos",
    "ddos":            "ddos",

    # brute-force / password
    "FTP-Patator":     "password",
    "SSH-Patator":     "password",
    "password":        "password",
    "Web Attack \ufffd Brute Force": "bruteforce",
    "bruteforce":      "bruteforce",

    # injection / xss
    "Web Attack \ufffd Sql Injection": "injection",
    "injection":       "injection",
    "Web Attack \ufffd XSS":          "xss",
    "xss":             "xss",
    # infiltration / backdoor
    "Infiltration":    "infiltration",
    "backdoor":        "infiltration",

    # bot / other
    "Bot":             "bot",
    "bot":             "bot",

    # other dataset–only classes
    "ransomware":      "ransomware",
    "mitm":            "mitm",
}


def normalize_labels(df, class_col):
    # Map every label, and send anything unmapped to 'other'
    return (
        df[class_col]
        .map(LABEL_MAPPING)           # map known → unified
        .fillna("other")              # everything else → other
    )
