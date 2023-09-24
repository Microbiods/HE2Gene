
import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("filenames", type=str, nargs="+")
parser.add_argument("--gene", type=str, nargs="?", default="FASN")
parser.add_argument("--figroot", default="visualize/", type=str)


argcomplete.autocomplete(parser)
args = parser.parse_args()

# print( args.filenames)
method = args.filenames[0].split("/")[0]

import sys
sys.path.append(".")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import pathlib
import scipy.stats
from sklearn.decomposition import PCA

def latexify():
    import matplotlib
    params = {'backend': 'pdf',
          'axes.titlesize':  8,
          'axes.labelsize':  8,
          'font.size':       8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          # 'text.usetex': True,
          'font.family': 'DejaVu Serif',
          'font.serif': 'Computer Modern',
    }
    matplotlib.rcParams.update(params)

latexify()
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(1.00, 0.50))
cmap_bin = plt.get_cmap("binary")
plt.scatter([-1], [-1], color=cmap_bin(1 - 1e-5), label="Tumor",  s=10, linewidth=0.2, edgecolors='black')
plt.scatter([-1], [-1], color=cmap_bin(1e-5),     label="Normal", s=10, linewidth=0.2, edgecolors="black")
plt.axis([0, 1, 0, 1])
plt.gca().axis("off")
plt.gca().set_aspect("equal")
plt.xticks([])
plt.yticks([])
plt.legend(loc="center", frameon=False)
# delete frame of legend

plt.tight_layout()
fig.savefig(args.figroot + "tumor_legend.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.25, 0.75))
norm = matplotlib.colors.Normalize(vmin=-3, vmax=3)
cb = matplotlib.colorbar.ColorbarBase(plt.gca(), cmap=cmap,
                                      norm=norm,
                                      orientation='horizontal')
cb.set_ticks(range(-3, 4))
cb.set_label("zscore")
plt.tight_layout()
fig.savefig(args.figroot + "colorbar.pdf")
plt.close(fig)

task = None
gene_names = None
patient = []
section = []
tumor = []
count = []
pred = []
pixel = []

for f in args.filenames:
    npz = np.load(f)

    # if task is not None:
    #     assert(task == data["task"])
    # task = data["task"]
    print(npz.files)

    data = {}
    select_patient = npz['patients']
    select_section = npz['sections']
    
    data["patients"] = npz["patients"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    data["sections"] = npz["sections"][np.where((select_patient=='BC23508') & (select_section=='D2'))]

    if method == 'his2st' or method == 'his2gene': 
        select_patient = npz['patients'][:, 0]
        select_section = npz['sections'][:, 0]
        data["patients"] = npz["patients"][:, 0][np.where((select_patient=='BC23508') & (select_section=='D2'))]
        data["sections"] = npz["sections"][:, 0][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    else:
        select_patient = npz['patients']
        select_section = npz['sections']
        data["patients"] = npz["patients"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
        data["sections"] = npz["sections"][np.where((select_patient=='BC23508') & (select_section=='D2'))]


    if method == 'his2st' or method == 'his2gene':  
        # print(([np.where((select_patient=='BC23508') & (select_section=='D2'))]))
        data["annotations"] = np.array(npz["annotations"]).reshape(-1, 1)[np.where((select_patient=='BC23508') & (select_section=='D2'))]
    else:
        # print(npz["annotations"].shape)
        data["annotations"] = npz["annotations"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    
    data["real_counts"] = npz["real_counts"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    

    if method == 'he2gene_base':
        data["predictions"] = npz["grid_pred_counts"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    elif method == 'he2gene_base_spat':
        data["predictions"] = npz["tta_pred_counts"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    else:
        data["predictions"] = npz["pred_counts"][np.where((select_patient=='BC23508') & (select_section=='D2'))]

    print(data["predictions"].shape)


    data["pixels"] = npz["pixels"][np.where((select_patient=='BC23508') & (select_section=='D2'))]
    data["traget_genes"] = npz["traget_genes"]

    if gene_names is not None:
        assert((gene_names == data["traget_genes"]).all())
    gene_names = data["traget_genes"]

    patient.append(data["patients"])
    section.append(data["sections"])
    tumor.append(data["annotations"])
    count.append(data["real_counts"])
    pred.append(data["predictions"])
    pixel.append(data["pixels"])

patient = np.concatenate(patient)
section = np.concatenate(section)
ps = list(zip(patient, section))
tumor = np.concatenate(tumor)
count = np.concatenate(count)
pred = np.concatenate(pred)
pixel = np.concatenate(pixel)


if pred.shape[0] == count.shape[0] and pred.shape[1] != count.shape[1]:
    # For total count prediction
    count = np.sum(count, 1, keepdims=True)
    gene_names = ["count"]

# if task == "gene":
    # pca = PCA(n_components=3)
    # pca.fit(count)
    # print("PCA Explained Variance: {}".format(pca.explained_variance_ratio_))
    # print(count.shape)
    # c = pca.transform(count)
    # p = pca.transform(pred)

index = np.argmax(gene_names == args.gene)
c = count[:, index]
p = pred[:, index]
print(gene_names[index])

for (patient, section) in sorted(set(ps)):
    tol = 1e-5
    mask = np.array([((p == patient) and (s == section)) for (p, s) in ps])
    margin = 112 + 50
    xmin, xmax, ymin, ymax = min(pixel[mask, 0]), max(pixel[mask, 0]), min(pixel[mask, 1]), max(pixel[mask, 1])

    # TODO: check aspect ratio
    # TODO: remove box/axis 
    print(patient, section, flush=True)
    # image_filename = os.path.join(patient, "{}_{}.tif".format(patient, section))
    # image = plt.imread(image_filename)

    xsize = xmax - xmin + 2 * margin
    ysize = ymax - ymin + 2 * margin
    figsize = (0.00017 * xsize, 0.00017 * ysize)

    fig = plt.figure(figsize=figsize)
    # plt.imshow(image, aspect="equal", interpolation="nearest")
    fig.patch.set_visible(False)
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    x = xmin + 300
    y = ymax - 300
    plt.gca().annotate("1 mm", (x + 743, y - 200), fontsize=5, ha="center")
    plt.plot([x, x + 1486],        [y, y],             color="black", linewidth=0.3)
    plt.plot([x, x],               [y - 100, y + 100], color="black", linewidth=0.3)
    plt.plot([x + 1486, x + 1486], [y - 100, y + 100], color="black", linewidth=0.3)
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig("{}{}_{}.jpg".format(args.figroot, patient, section),  dpi=600)
    fig.savefig("{}{}_{}.pdf".format(args.figroot, patient, section),  dpi=600)
    plt.close(fig)

    # assert(task == "gene")

    coef, _ = scipy.stats.pearsonr(c[mask], p[mask])
    print(coef)
    coef, _ = scipy.stats.pearsonr(c, p)
    print(coef)
    print("tumor: ", sum(tumor[mask, 0]) / sum(mask))

    value = c[mask]
    value = ((value - value.mean(0)) / (3 * value.std(0) + tol) + 0.5).clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=10, linewidth=0, edgecolors="none")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig("{}{}_{}_{}.pdf".format(args.figroot, patient, section, args.gene))
    plt.close(fig)

    value = p[mask]
    value = ((value - value.mean(0)) / (3 * value.std(0) + tol) + 0.5).clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap, value)), s=10, linewidth=0, edgecolors="none")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig("{}{}_{}_{}_{}_{}.pdf".format(args.figroot, patient, section, args.gene, "pred", method))
    plt.close(fig)

    value = tumor[mask, 0].clip(tol, 1 - tol)
    fig = plt.figure(figsize=figsize)
    cmap_bin = plt.get_cmap("binary")
    plt.scatter(pixel[mask, 0], pixel[mask, 1], color=list(map(cmap_bin, value)), s=10, linewidth=0.2, edgecolors="k")
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")
    plt.xticks([])
    plt.yticks([])
    plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig.savefig("{}{}_{}_{}.pdf".format(args.figroot, patient, section, "tumor"))
    plt.close(fig)



# python visualize.py predictions.npz --gene A1BG
