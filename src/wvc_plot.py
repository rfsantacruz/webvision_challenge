import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-white')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 16


def plot_tb_files(title, legends, files):
    # Plot
    fig, ax = plt.subplots(1)
    for file, legend in zip(files, legends):
        df = pd.read_csv(file)
        x = df['Step'].values
        y = df['Value'].values
        ax.plot(x, y, label=legend, linewidth=2)

    # Configure
    ax.legend(loc='lower right', ncol=2, fancybox=True, framealpha=0.5)
    ax.set_xlabel('epochs')
    ax.set_ylabel('value')
    ax.grid(linestyle='dashed', which='major', c='darkgray')
    ax.grid(linestyle='dotted', which='minor', c='lightgray')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Plotting tool")
    parser.add_argument('title', type=str, help="Plot Title")
    parser.add_argument('-legends', type=str, nargs='*', help='List of legends.')
    parser.add_argument('-files', type=str, nargs='*', help='List of files.')
    args = parser.parse_args()
    plot_tb_files(args.title, args.legends, args.files)