import re
import networkx as nx
import csv

def find_related_nodes(node, graph):
    ancestors = nx.ancestors(graph, node)
    descendants = nx.descendants(graph, node)
    return ancestors.union(descendants)




datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
               "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
for dataset in datasets:
    with open(f'Bif_File/{dataset}.bif', 'r') as file:
        bif_content = file.read()
    nodes = re.findall(r'variable\s+(\w+)\s+\{', bif_content)


    causal_relations = re.findall(r'probability\s*\(\s*([^\)]+)\s*\)\s*\{', bif_content)


    relations = []
    for relation in causal_relations:
        involved_nodes = relation.split('|')
        if len(involved_nodes) == 2:
            parent_nodes = involved_nodes[1].split(',')
            child_node = involved_nodes[0].strip()
            for parent in parent_nodes:
                relations.append((parent.strip(), child_node))
        else:
            # ����ǵ����ڵ�ĸ��ʣ����� A��
            relations.append((involved_nodes[0].strip(), None))

    # nodes, relations  # ����ڵ��б�������ϵ�б�



    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([(parent, child) for parent, child in relations if child])




    csv_file_path = f'Prompt2/{dataset}.csv'
    with open(csv_file_path, 'w', newline='',) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Relationship', 'Number'])

        output_nodes = []
        for node_a in nodes:
            related_to_a = find_related_nodes(node_a, graph)

            for node_b in nodes:
                if node_b != node_a:
                    related_to_b = find_related_nodes(node_b, graph)
                    if node_b in related_to_a:
                        csvwriter.writerow([f"Is {node_a} and {node_b} related?", 1])

                    if node_b not in related_to_a:
                        for node_c in nodes:
                            if node_c != node_a and node_c != node_b:

                                if node_c in related_to_b and node_c in related_to_a:
                                    output_nodes.append((node_a, node_b, node_c))
                                    csvwriter.writerow([f"If {node_a} and {node_b} are not directly related, then {node_a} and {node_b} are conditionally related under {node_c}. Under this condition, is {node_a} and {node_b} related?", 1])
                                else:
                                    csvwriter.writerow([f"If {node_a} and {node_b} are not directly related, then {node_a} and {node_b} are not conditionally related under {node_c}. Under this condition, is {node_a} and {node_b} related?", 0])


    print(output_nodes)
