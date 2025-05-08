# cic_ids_2017_classes = ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration',
#        'Web Attack � Brute Force', 'Web Attack � XSS',
#        'Web Attack � Sql Injection', None, 'FTP-Patator', 'SSH-Patator',
#        'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
#        'Heartbleed']


# cic_ton_iot_classes = ['Benign', 'mitm', 'scanning', 'dos', 'ddos', 'injection',
#        'password', 'backdoor', 'ransomware', 'xss']

# 1. Define your unified label mapping
LABEL_MAPPING_GROUPED = {
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

    "Heartbleed": "heartbleed"
}

LABEL_MAPPING_CLEANED_NAMES = {
    "BENIGN":          "benign",
    "Benign":          "benign",

    "PortScan":        "portscan",

    "scanning":        "scanning",

    "DoS Hulk":        "dos_hulk",
    "DoS slowloris":   "dos_slowloris",
    "DoS Slowhttptest": "dos_slowhttptest",
    "DoS GoldenEye":   "dos_goldenEye",
    "dos":             "dos",

    "DDoS":            "ddos",
    "ddos":            "ddos",

    "FTP-Patator":     "ftp-patator",
    "SSH-Patator":     "ssh-patator",

    "password":        "password",

    "Web Attack \ufffd Brute Force": "bruteforce",
    "bruteforce":      "bruteforce",

    "Web Attack \ufffd Sql Injection": "sql_injection",

    "injection":       "injection",

    "Web Attack \ufffd XSS":          "xss",
    "xss":             "xss",

    "Infiltration":    "infiltration",

    "backdoor":        "backdoor",

    "Bot":             "bot",
    "bot":             "bot",

    "ransomware":      "ransomware",

    "mitm":            "mitm",

    "Heartbleed": "heartbleed"
}


def normalize_labels(df, class_col):
    # Map every label, and send anything unmapped to 'other'
    return (
        df[class_col]
        # .map(LABEL_MAPPING_CLEANED_NAMES)           # map cleanted
        .map(LABEL_MAPPING_GROUPED)           # map known → unified
        .fillna("other")              # everything else → other
    )
