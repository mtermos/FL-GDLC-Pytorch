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
    "benign":          "benign",

    # scanning / portscan
    "PortScan":        "scanning",
    "portscan":        "scanning",
    "scanning":        "scanning",
    "Reconnaissance":   "scanning",
    "reconnaissance":   "scanning",

    # dos variants
    "DoS Hulk":        "dos",
    "dos_hulk":        "dos_hulk",
    "DoS slowloris":   "dos",
    "dos_slowloris":   "dos_slowloris",
    "DoS Slowhttptest": "dos",
    "dos_slowhttptest": "dos_slowhttptest",
    "DoS GoldenEye":   "dos",
    "dos_goldenEye":   "dos_goldenEye",
    "dos":             "dos",
    "DoS":             "dos",
    "Syn":              "dos",
    "syn":              "syn",
    "MSSQL":            "dos",
    "mssql":            "mssql",

    # ddos
    "DDoS":            "ddos",
    "ddos":            "ddos",

    "TFTP":            "tftp",
    "tftp":            "tftp",
    "DrDoS_SNMP":       "drdos_snmp",
    "drdos_snmp":       "drdos_snmp",
    "DrDoS_DNS":        "drdos_dns",
    "drdos_dns":        "drdos_dns",
    "DrDoS_MSSQL":      "drdos_mssql",
    "drdos_mssql":      "drdos_mssql",
    "DrDoS_NetBIOS":    "drdos_netbios",
    "drdos_netbios":    "drdos_netbios",
    "UDP":              "udp",
    "udp":              "udp",
    "DrDoS_UDP":        "drdos_udp",
    "drdos_udp":        "drdos_udp",
    "DrDoS_SSDP":       "drdos_ssdp",
    "drdos_ssdp":       "drdos_ssdp",
    "DrDoS_LDAP":       "drdos_ldap",
    "drdos_ldap":       "drdos_ldap",
    "LDAP":             "ldap",
    "ldap":             "ldap",
    "DrDoS_NTP":        "drdos_ntp",
    "drdos_ntp":        "drdos_ntp",
    "UDP_lag":          "udp_lag",
    "udp_lag":          "udp_lag",
    "Portmap":          "portmap",
    "portmap":          "portmap",
    "UDPLag":           "udplag",
    "udplag":           "udplag",
    "WebDDoS":          "webddos",
    "webddos":          "webddos",

    "NetBIOS":          "netbios",
    "netbios":          "netbios",

    # brute-force / password
    "FTP-Patator":     "password",
    "ftp-patator":     "password",
    "SSH-Patator":     "password",
    "ssh-patator":     "password",
    "password":        "password",
    "Web Attack \ufffd Brute Force": "bruteforce",
    "bruteforce":      "bruteforce",

    # injection / xss
    "Web Attack \ufffd Sql Injection": "injection",
    "injection":       "injection",
    "sql_injection":       "sql_injection",
    "Web Attack \ufffd XSS":          "xss",
    "xss":             "xss",
    # infiltration / backdoor
    "Infiltration":    "infiltration",
    "infiltration":    "infiltration",
    "backdoor":        "infiltration",
    "Theft":            "infiltration",
    "theft":            "infiltration",
    # "theft":            "theft",

    # bot / other
    "Bot":             "bot",
    "bot":             "bot",

    # other dataset–only classes
    "ransomware":      "ransomware",
    "mitm":            "mitm",

    "Heartbleed": "heartbleed",
    "heartbleed": "heartbleed"
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
    "DoS":             "dos",

    "DDoS":            "ddos",
    "ddos":            "ddos",
    "TFTP":            "tftp",
    "Syn":              "syn",
    "MSSQL":            "mssql",
    "DrDoS_SNMP":       "drdos_snmp",
    "DrDoS_DNS":        "drdos_dns",
    "DrDoS_MSSQL":      "drdos_mssql",
    "DrDoS_NetBIOS":    "drdos_netbios",
    "UDP":              "udp",
    "NetBIOS":          "netbios",
    "DrDoS_UDP":        "drdos_udp",
    "DrDoS_SSDP":       "drdos_ssdp",
    "DrDoS_LDAP":       "drdos_ldap",
    "LDAP":             "ldap",
    "DrDoS_NTP":        "drdos_ntp",
    "UDP_lag":          "udp_lag",
    "Portmap":          "portmap",
    "UDPLag":           "udplag",
    "WebDDoS":          "webddos",
    "Theft":            "theft",

    "Reconnaissance":   "reconnaissance",

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

CATEGORY_TO_CLASSES = {
    "benign": [
        "BENIGN",
        "Benign",
        "benign",
    ],
    "scanning": [
        "PortScan",
        "portscan",
        "scanning",
        "Reconnaissance",
        "reconnaissance",
    ],
    "dos": [
        "dos",
        "DoS",
        "DoS Hulk",
        "dos_hulk",
        "DoS slowloris",
        "dos_slowloris",
        "DoS Slowhttptest",
        "dos_slowhttptest",
        "DoS GoldenEye",
        "dos_goldenEye",
        "Syn",              # SYN‐flood (single‐source DoS)
        "syn",              # SYN‐flood (single‐source DoS)
        "MSSQL",            # plain MSSQL‐floods → DoS
        "mssql",            # plain MSSQL‐floods → DoS
    ],
    "ddos": [
        "DDoS",
        "ddos",
        "DrDoS_SNMP",
        "drdos_snmp",
        "DrDoS_DNS",
        "drdos_dns",
        "DrDoS_MSSQL",
        "drdos_mssql",
        "DrDoS_NetBIOS",
        "drdos_netbios",
        "DrDoS_UDP",
        "drdos_udp",
        "DrDoS_SSDP",
        "drdos_ssdp",
        "DrDoS_LDAP",
        "drdos_ldap",
        "DrDoS_NTP",
        "drdos_ntp",
        "UDP",
        "udp",
        "UDP_lag",
        "udp_lag",
        "UDPLag",
        "udplag",
        "WebDDoS",
        "webddos",
        "TFTP",             # TFTP‐amplification
        "tftp",             # TFTP‐amplification
        "LDAP",             # LDAP‐amplification
        "ldap",             # LDAP‐amplification
        "netbios",
        "NetBIOS",          # treat probing for NetBIOS as reconnaissance
        "Portmap",
        "portmap",
    ],
    "password": [
        "FTP-Patator",
        "ftp-patator",
        "SSH-Patator",
        "ssh-patator",
        "password",
    ],
    "bruteforce": [
        "Web Attack \ufffd Brute Force",
        "bruteforce",
    ],
    "injection": [
        "Web Attack \ufffd Sql Injection",
        "injection",
        "sql_injection",
    ],
    "xss": [
        "Web Attack \ufffd XSS",
        "xss",
    ],
    "infiltration": [
        "Infiltration",
        "infiltration",
        "backdoor",
        "Theft",            # data exfiltration → treat as infiltration
        "theft",            # data exfiltration → treat as infiltration
        "ransomware",       # compromise/ransomware behavior
        "mitm",             # man‐in‐the‐middle belongs to infiltration
        "Heartbleed",       # fits under “infiltration”/exploit
        "heartbleed",       # fits under “infiltration”/exploit
    ],
    "bot": [
        "Bot",
        "bot",
    ],
}


# 2) Turn that “category → [raw‐labels]” structure
#    into a flat mapping “raw_label → category”:
LABEL_MAPPING_10CATS = {
    raw_label: category
    for category, raw_list in CATEGORY_TO_CLASSES.items()
    for raw_label in raw_list
}


def normalize_labels(df, class_col):
    # 1. Grab all unique values from your class column
    unique_labels = set(df[class_col].unique())

    # 2. Grab all the keys that your mapping already covers
    mapped_keys = set(LABEL_MAPPING_GROUPED.keys())

    # 3. Any label that isn’t in mapped_keys will end up as "other"
    missing_labels = unique_labels - mapped_keys

    print("Labels not yet in LABEL_MAPPING_GROUPED:")
    print(sorted(missing_labels))

    # Map every label, and send anything unmapped to 'other'
    return (
        df[class_col]
        # .map(LABEL_MAPPING_CLEANED_NAMES)           # map cleanted
        .map(LABEL_MAPPING_10CATS)           # map known → unified
        .fillna("other")              # everything else → other
    )
