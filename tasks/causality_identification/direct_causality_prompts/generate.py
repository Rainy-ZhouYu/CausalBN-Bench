import subprocess

datasets = ["asia", "cancer", "earthquake", "sachs", "survey", "alarm", "barley", "child",
            "insurance", "mildew", "water", "hailfinder", "hepar2", "win95pts",
            "andes", "diabetes", "link", "munin1", "pathfinder", "pigs",
            "munin", "munin2","munin3","munin4"]
for dataset in datasets:
    subprocess.run(['python', f'generate_{dataset}.py'])