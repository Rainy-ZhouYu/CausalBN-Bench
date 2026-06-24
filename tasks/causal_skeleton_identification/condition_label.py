import re
import networkx as nx
import csv
# 从BIF内容中提取节点名称
def find_related_nodes(node, graph):
    """找到与给定节点相关的所有节点（直接或间接因果关系）"""
    ancestors = nx.ancestors(graph, node)
    descendants = nx.descendants(graph, node)
    return ancestors.union(descendants)




datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
               "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts"]
for dataset in datasets:
    with open(f'Bif_File/{dataset}.bif', 'r') as file:
        bif_content = file.read()
    nodes = re.findall(r'variable\s+(\w+)\s+\{', bif_content)

    # 从BIF内容中提取因果关系，即从每个"probability"块中提取
    causal_relations = re.findall(r'probability\s*\(\s*([^\)]+)\s*\)\s*\{', bif_content)

    # 分析每个probability块中的节点和它们的关系
    relations = []
    for relation in causal_relations:
        involved_nodes = relation.split('|')
        if len(involved_nodes) == 2:
            # 如果存在条件关系（例如 A | B,C）
            parent_nodes = involved_nodes[1].split(',')
            child_node = involved_nodes[0].strip()
            for parent in parent_nodes:
                relations.append((parent.strip(), child_node))
        else:
            # 如果是单独节点的概率（例如 A）
            relations.append((involved_nodes[0].strip(), None))

    # nodes, relations  # 输出节点列表和因果关系列表



    # 由于环境中缺少pgmpy库，我将使用之前提取的节点和关系列表来实现逻辑。
    # 这里将使用networkx库来分析图形结构，但请注意这是一种近似方法，并非严格的贝叶斯网络分析。

    # 创建一个有向图来表示因果骨架
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([(parent, child) for parent, child in relations if child])



    # 对于每个节点 A，找出与其相关的节点，并进行相关性判断
    csv_file_path = f'Label/{dataset}.csv'
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
                        csvwriter.writerow([f"{node_a} and {node_b} are related", 1])
                    # 如果A和B不相关
                    if node_b not in related_to_a:
                        for node_c in nodes:
                            if node_c != node_a and node_c != node_b:
                                # 如果节点C与节点B相关，且节点C也与节点A相关
                                if node_c in related_to_b and node_c in related_to_a:
                                    output_nodes.append((node_a, node_b, node_c))
                                    csvwriter.writerow([f"{node_a} and {node_b} are conditionally related under {node_c}", 1])
                                else:
                                    csvwriter.writerow([f"{node_a} and {node_b} are not conditionally related under {node_c}", 0])
    # 找到符合条件的节点组合后跳出循环

    print(output_nodes)  # 输出满足条件的节点组合列表
