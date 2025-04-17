# Complex Network Measures Selection, 4 types:
# type_1: all classical centralities + Local Modular Centralities + Comm Centrality + K-core + K-truss \\
# type_2: all classical centralities + Global Modular Centralities + Modularity Vitality + K-core + K-truss \\
# type_3: Betweenness and PageRank + Local Modular Centralities + Comm Centrality + K-core + K-truss \\
# type_4: Betweenness and PageRank + Global Modular Centralities + Modularity Vitality + K-core + K-truss \\

'''

This class constructor initializes an object to hold information about the dataset's structure and relevant features for analysis,
including which columns are used for specific types of data, the format for timestamps, which columns to drop during preprocessing,
and specific features or measures of interest for network analysis. This approach facilitates the management of dataset attributes
and preprocessing steps, making it easier to apply consistent handling across different datasets or analysis scripts.

At the end of the file we provided an array with the datasets we used in this experiment
'''


class DatasetInfo:
    def __init__(
            self,

            # The name of the dataset.
            name,

            # Column name for source IP addresses in the dataset.
            src_ip_col="Src IP",

            # Column name for destination IP addresses.
            dst_ip_col="Dst IP",

            # Column name for identifying unique network flows.
            flow_id_col="Flow ID",

            # Column name for timestamps of network events.
            timestamp_col="Timestamp",

            # Format of the timestamp data.
            timestamp_format="%d/%m/%Y %I:%M:%S %p",

            # Column name for the label or outcome (e.g., "normal" or "attack").
            label_col="Label",

            # Column name for the class of attack or normal behavior.
            class_col="Attack",
            class_num_col="Class",

            # Columns to be dropped from the dataset during preprocessing.
            drop_columns=["Flow ID", "Src IP", "Dst IP",
                          "Timestamp", "Src Port", "Dst Port", "Attack"],
            drop_columns_w_class=["Flow ID", "Src IP", "Dst IP",
                                  "Timestamp", "Src Port", "Dst Port",  "Attack"],

            # Columns to be dropped from the dataset during preprocessing.
            weak_columns=[],
            # Classes with very small availability. To be dropped.
            low_classes=[],

            # List of complex network measures added to this dataset.
            cn_measures=[],


            # List of names used for these complex network.
            network_features=[]
    ):

        self.name = name
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        self.flow_id_col = flow_id_col
        self.timestamp_col = timestamp_col
        self.timestamp_format = timestamp_format
        self.label_col = label_col
        self.class_col = class_col
        self.class_num_col = class_num_col
        self.drop_columns = drop_columns
        self.weak_columns = weak_columns
        self.drop_columns_w_class = drop_columns_w_class
        self.cn_measures = cn_measures
        self.network_features = network_features


# cn_measures_type_1 = ["betweenness", "local_betweenness", "degree", "local_degree",
#                       "closeness", "pagerank", "local_pagerank",  "k_truss", "Comm"]
# network_features_type_1 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree',
#                            'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']

cn_measures_type_1 = ["betweenness", "local_betweenness", "degree", "local_degree",
                      "eigenvector", "closeness", "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"]
network_features_type_1 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree', 'src_eigenvector',
                           'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']

cn_measures_type_2 = ["betweenness", "global_betweenness", "degree", "global_degree",
                      "eigenvector", "closeness", "pagerank", "global_pagerank", "k_core", "k_truss", "mv"]
network_features_type_2 = ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_degree', 'dst_degree', 'src_global_degree', 'dst_global_degree', 'src_eigenvector',
                           'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv']

cn_measures_type_3 = ["betweenness", "local_betweenness",
                      "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"]
network_features_type_3 = ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_pagerank',
                           'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm']

cn_measures_type_4 = ["betweenness", "global_betweenness",
                      "pagerank", "global_pagerank", "k_core", "k_truss", "mv"]
network_features_type_4 = ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_pagerank',
                           'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv']

weak_columns = ['Flow Duration', 'Tot Bwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Pkts/s', 'Flow IAT Mean',
                'Flow IAT Max', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Active Mean', 'Idle Mean']


datasets = [
    # 0
    DatasetInfo("cic_ton_iot",
                cn_measures=cn_measures_type_2,
                network_features=network_features_type_2,
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Bwd Header Len', 'Tot Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                              'CWE Flag Count', 'Bwd IAT Tot', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Min', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Var', 'FIN Flag Cnt', 'Bwd IAT Mean', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Tot', 'PSH Flag Cnt', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    # 1
    DatasetInfo("cic_ids_2017",
                timestamp_format="mixed",
                cn_measures=cn_measures_type_1,
                network_features=network_features_type_1,
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
]
