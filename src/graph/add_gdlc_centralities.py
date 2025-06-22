from src.graph.graph_level_measures import compute_graph_properties
from src.graph.gdlc_types import gdlc_centrality_measures, gdlc_network_features
from src.graph.add_centralities import add_centralities


def add_gdlc_centralities(df, src_ip_col, dst_ip_col, G):
    gdlc_type, properties = get_gdlc_type(G)
    centrality_measures = gdlc_centrality_measures[gdlc_type-1]
    network_features = gdlc_network_features[gdlc_type-1]

    add_centralities(df, new_path=None, graph_path=None, src_ip_col=src_ip_col, dst_ip_col=dst_ip_col,
                     cn_measures=centrality_measures, network_features=network_features, G=G)

    return network_features, gdlc_type, properties


def get_gdlc_type(G):
    graph_properties = compute_graph_properties(G)

    density = graph_properties["density"]
    transitivity = graph_properties["transitivity"]
    mixing_param = graph_properties["mixing_parameter"]


    properties = {
        "density": density,
        "mixing_param": mixing_param,
    }
    print("==============================")
    print(f"====> properties: {properties}")
    if density < 0.1:
        is_low_density = True
    else:
        is_low_density = False

    # if transitivity < 0.01:
    #     is_low_transitivity = True
    # else:
    #     is_low_transitivity = False

    if mixing_param < 0.1:
        is_low_mixing = True
    else:
        is_low_mixing = False

    # while True:
    #     response = input("Is this density considered low? (y/n): ").lower()
    #     if response in ['y', 'n']:
    #         is_low_density = response == 'y'
    #         break
    #     print("Please answer with 'y' or 'n'")

    # print(f"The transitivity is {transitivity:.3f}")
    # while True:
    #     response = input(
    #         "Is this transitivity considered low? (y/n): ").lower()
    #     if response in ['y', 'n']:
    #         is_low_transitivity = response == 'y'
    #         break
    #     print("Please answer with 'y' or 'n'")

    # print(f"The mixing parameter is {mixing_param:.3f}")
    # while True:
    #     response = input(
    #         "Is this mixing parameter considered low? (y/n): ").lower()
    #     if response in ['y', 'n']:
    #         is_low_mixing = response == 'y'
    #         break
    #     print("Please answer with 'y' or 'n'")

    # if is_low_density and is_low_transitivity and not is_low_mixing:
    #     gdlc_type = 1
    # if is_low_density and is_low_transitivity and is_low_mixing:
    #     gdlc_type = 2
    # if not is_low_density and not is_low_transitivity and not is_low_mixing:
    #     gdlc_type = 3
    # if not is_low_density and not is_low_transitivity and is_low_mixing:
    #     gdlc_type = 4

    if is_low_density and not is_low_mixing:
        gdlc_type = 1
    if is_low_density and is_low_mixing:
        gdlc_type = 2
    if not is_low_density and not is_low_mixing:
        gdlc_type = 3
    if not is_low_density and is_low_mixing:
        gdlc_type = 4

    print(f"==>> gdlc_type: {gdlc_type}")

    return gdlc_type, properties
