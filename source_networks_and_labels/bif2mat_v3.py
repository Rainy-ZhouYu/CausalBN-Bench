import bnlearn as bn

# 导入.bif文件
model = bn.import_DAG('alarm.bif')

# 打印模型的结构
bn.print_CPD(model)
