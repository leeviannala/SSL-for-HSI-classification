from main import load_datasets

def calc(ds, sp, datasets_root):
    load_datasets(ds, datasets_root, sp)

if __name__ == "__main__":
    datasets_root = '/mnt/data/leevi/'
    # datasets = ['KSC', 'PaviaU', 'Botswana', 'IndianPines']
    datasets = ['Botswana', 'IndianPines']
    sample_percentages = [0.01, 0.05, 0.10, 1, 5, 10]
    for ds in datasets:
        for sp in sample_percentages:
            try:
                calc(ds, sp, datasets_root)
            except ValueError:
                pass