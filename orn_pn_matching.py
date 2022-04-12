import numpy as np
import plotting_help_py37 as ph

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os, random, math, copy


# helper function for analyzing seq data

def get_glomtypes_from_seqss(seqss, datatype_orn):
    if datatype_orn:
        types_ = np.array(seqss.obs.all_types)
        types_[seqss.obs.MARS_matched == 'MARS_2'] = 'DL1'
        types_[seqss.obs.MARS_matched == 'MARS_3'] = 'DC1'
        types_[seqss.obs.MARS_matched == 'MARS_5'] = 'VM6'
        types_[seqss.obs.all_types == 'DA41'] = 'DA4l'
        return types_
    else:
        types_ = np.array(seqss.obs.PN_type)
        types_[seqss.obs.PN_type == 'DA1_fru+'] = 'DA1'
        types_[seqss.obs.PN_type == 'DA1_fru-'] = 'DA1'
        types_[seqss.obs.PN_type == 'VM7v(1)'] = 'VM7v'
        # types_[seqss.obs.PN_type == 'MARS_5'] = 'DA3'
        # types_[seqss.obs.PN_type == 'MARS_27'] = 'DM5'
        # types_[seqss.obs.PN_type == 'MARS_7'] = 'VA3'
        types_[seqss.obs.PN_type == 'VM7 or VM5v #1'] = 'VM7'
        types_[seqss.obs.PN_type == 'VM7 or VM5v #2'] = 'VM5v'
        types_[seqss.obs.PN_type == 'DA41'] = 'DA4l'
        types_[seqss.obs.PN_type == 'MARS_2'] = 'DA2'
        types_[seqss.obs.PN_type == 'MARS_9'] = 'VA7m'
        types_[seqss.obs.PN_type == 'MARS_25'] = 'VM1'
        return types_

def get_genename_from_seqss(seqss, gene):
    names = []
    n = len(gene)
    for gene_seq in seqss.var_names:
        if len(gene_seq) >= 2:
            if gene_seq[:n].casefold() == gene.casefold():
                names.append(gene_seq)
    return names

def get_CSMs(genes, infunc_print=True):
    csms = []
    for gene in genes:
        if gene in CSMs:
            if infunc_print:
                print(gene)
            else:
                csms.append(gene)
    if not infunc_print:
        return csms

def get_glomid(gloms, infunc_print=True):
    if infunc_print:
        for glom in gloms:
            print('%s: %i %i' % (glom, glomid_list.index('%s_R' % glom), glomid_list.index('%s_L' % glom)))
    else:
        ids = []
        for glom in gloms:
            glom_ = ('%s_R' % glom)
            if glom_ in glomid_list:
                ids.append(glomid_list.index(glom_))
            else:
                ids.append(-1)
            glom_ = ('%s_L' % glom)
            if glom_ in glomid_list:
                ids.append(glomid_list.index(glom_))
            else:
                ids.append(-1)
        return ids

def get_glomcentroid(mm, glom_='DA1', direction='z'):
    factor = 0.6221
    if direction == 'z':
        glom_ids = get_glomid([glom_], infunc_print=False)
        zs = []
        for glom_id in glom_ids:
            glom_xyz = (mm == glom_id+1)
            a = np.sum(glom_xyz, axis=(0,1))
            b = np.arange(len(a))
            zs.append(np.mean(a * b) / np.mean(a))
    return np.mean(zs)*factor


def mark_CSMs(genes,):
    genes_intersection = []
    for gene_ in genes:
        gene_toappend = gene_
        if gene_ in CSMs:
            gene_toappend = gene_ + '*'
        genes_intersection.append(gene_toappend)
    return genes_intersection

def CSMs_only(genes, infunc_print=False):
    genes_output = []
    for gene_ in genes:
        if gene_ in CSMs:
            genes_output.append(gene_)
    if infunc_print:
        print(', '.join(genes_output))
    return genes_output

def score_genes_1glom(genes_expr, stain_1glom, score_method='binary_linear'):
    if score_method == 'binary_linear':
        if stain_1glom:
            genes_score = genes_expr * (1)
        else:
            genes_score = genes_expr * (-1)
    elif score_method == 'pospun_negthr':
        if stain_1glom:
            genes_score = genes_expr * (2)
            genes_score[genes_score > 8] = 8
        else:
            genes_score = genes_expr * (-1)
            genes_score[genes_score < -4] = -4
    else:
        print('Error: does not have this type of score_method')
        return []

    return genes_score

def score_genes_gloms(seqss, stain_dict, datatype_orn=True, score_method='binary_linear'):
    all_glom_types = get_glomtypes_from_seqss(seqss, datatype_orn)

    yss = []            # 2D matrix, first dimension contains glomeruli, second contains genes
    gloms = []
    for glom_ in np.unique(all_glom_types):
        if glom_ != 'NaN':
            seqss_1glom = seqss[all_glom_types == glom_]
            genes_expr_1glom = np.array(np.nanmean(seqss_1glom.X, axis=0))
            if glom_ in stain_dict:
                stain_1glom = stain_dict[glom_]
                genes_score = score_genes_1glom(genes_expr_1glom, stain_1glom, score_method=score_method)
                yss.append(genes_score)
                gloms.append(glom_)
    yss = np.array(yss)

    # normalize positive and negative scores, separately, by glomeruli number
    idxs_posvalue = np.array([stain_dict[glom_] > 0 for glom_ in gloms])
    yss[idxs_posvalue, :] = yss[idxs_posvalue, :] / np.sum(idxs_posvalue)
    idxs_negvalue = ~idxs_posvalue
    yss[idxs_negvalue, :] = yss[idxs_negvalue, :] / np.sum(idxs_negvalue)

    # stats
    ysmean = np.nanmean(yss, axis=0)
    ysmean_sorted = np.sort(ysmean)[::-1]
    ys_std_sorted = np.nanstd(yss, axis=0)[np.argsort(ysmean)[::-1]]
    var_names_sorted = list(seqss.var_names[np.argsort(ysmean)[::-1]])

    return np.array(ysmean_sorted), np.array(ys_std_sorted), var_names_sorted

def intersect_genecandidates(vns1, vns2, top=True, n=50, infunc_print=True, mark_csms=True):
    list1 = vns1[:n] if top else vns1[-n:]
    if len(vns2):
        list2 = vns2[:n] if top else vns2[-n:]
    else:
        list2 = list1
    s_ = 'positive' if top else 'negative'
    genes_ = list(set(list1).intersection(list2))
    genes = mark_CSMs(genes_) if mark_csms else genes_

    if infunc_print:
        print('The intersection (%s) are:' % s_)
        print(genes)
    else:
        return genes

def print_intersect(vns1, vns2, ns=[100,10]):
    # attraction
    a1 = intersect_genecandidates(vns1, vns2, top=True, n=ns[0], infunc_print=False, mark_csms=False)
    a2 = intersect_genecandidates(vns1, [], top=True, n=ns[1], infunc_print=False, mark_csms=False)
    a3 = intersect_genecandidates(vns2, [], top=True, n=ns[1], infunc_print=False, mark_csms=False)
    genes_intersection = mark_CSMs(a1)
    print('Attraction candidates\n\tintersection (top %i): %s\n\tCSMs (top %i): %s' %
          (ns[0], ', '.join(genes_intersection), ns[1], ', '.join(mark_CSMs(get_CSMs(a2 + a3, infunc_print=False)))))

    # repulsion
    a1_ = intersect_genecandidates(vns1, vns2, top=False, n=ns[0], infunc_print=False, mark_csms=False)
    a2 = intersect_genecandidates(vns1, [], top=False, n=ns[1], infunc_print=False, mark_csms=False)
    a3 = intersect_genecandidates(vns2, [], top=False, n=ns[1], infunc_print=False, mark_csms=False)
    genes_intersection = mark_CSMs(a1_)
    print('Repulsion candidates\n\tintersection (top %i): %s\n\tCSMs-1 (top %i): %s\n\tCSMs-2 (top %i): %s' %
          (ns[0], ', '.join(genes_intersection), ns[1], ', '.join(mark_CSMs(get_CSMs(a2, infunc_print=False))),
           ns[1], ', '.join(mark_CSMs(get_CSMs(a3, infunc_print=False)))))

    return CSMs_only(a1+a1_)

# plot func

def plot_genecandidates_sorted(ys_mean, ys_std, ax=False, title='', genes=[], vns=[], fs=12, ms=4):
    ph.set_fontsize(fs)
    if not ax:
        ax = ph.large_ax([6, 4])
    xs = np.arange(len(ys_mean)) + 1
    ax.plot(xs, ys_mean, c='black', lw=1.5)
    ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, facecolor='black', edgecolor='none', alpha=.2)

    for gene in genes:
        x_ = vns.index(gene)
        ax.plot(x_+1, ys_mean[x_], marker='o', mec='none', mfc=ph.red, ms=ms, alpha=.7)
        # ax.text()

    xrange, yrange = xs[-1] - xs[0], ys_mean[0] - ys_mean[-1]
    xlim, ylim = [-.008 * xrange, xs[-1] + .008 * xrange], [ys_mean[-1] - .1 * yrange, ys_mean[0] + .1 * yrange]
    ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xlim=xlim, ylim=ylim)
    ax.set_xlabel('genes')
    ax.set_ylabel('score')
    if title:
        ax.set_title(title)

def plot_expr_afo_gloms_linearaxis(seqss, gene, stain_dict=None, datatype_orn=True, ax=False, score=False,
                                   fs=11, ms_stain=3, c_stain=ph.grey5, c=ph.orange, ms=5, title='', fs_title=16,
                                   show_axes=['left', 'bottom'], show_xticklabels=True, show_ylabel=True):
    # if stain_dict is available, use it to create the first half of the x axis
    if stain_dict is not None:
        stain_dict_sorted = dict(sorted(stain_dict.items(), reverse=True, key=lambda item: item[1]))
        gloms_stain = list(stain_dict_sorted.keys())
        stain_gloms = np.array(list(stain_dict_sorted.values()))
        gloms_seq = list(np.unique(get_glomtypes_from_seqss(seqss, datatype_orn)))
        gloms = copy.copy(gloms_stain)
        for glom_ in gloms_seq:
            if glom_ not in gloms_stain:
                gloms.append(glom_)
    else:
        gloms = list(np.unique(get_glomtypes_from_seqss(seqss, datatype_orn)))

    # calculate gene expression in each glom
    seqss_1gene = seqss[:, (seqss.var_names == gene)]
    ys_mean = []
    ys_std = []
    for glom_ in gloms:
        seqss_1gene1glom = seqss_1gene[get_glomtypes_from_seqss(seqss, datatype_orn) == glom_]
        if len(seqss_1gene1glom):
            ys_mean.append(np.nanmean(seqss_1gene1glom.X, axis=0))
            ys_std.append(np.nanstd(seqss_1gene1glom.X, axis=0))
        else:
            ys_mean.append(np.nan)
            ys_std.append(np.nan)

    # plot
    ph.set_fontsize(fs)
    if not ax:
        ax = ph.large_ax([9, 4])

    if stain_dict is not None:
        ax.plot(np.arange(len(gloms_stain)), stain_gloms, lw=.4, c=c_stain, marker='o', ms=ms_stain, mfc=c_stain)

    xs = np.arange(len(gloms))
    ys_mean = np.array(ys_mean, dtype=float).flatten()
    ys_std = np.array(ys_std, dtype=float).flatten()
    ax.errorbar(xs, ys_mean, ys_std, fmt='o', c=c, ms=ms, elinewidth=.5)

    if not title:
        csm_marker = '*' if gene in CSMs else ''
        title = '%s%s: %.3f' % (gene, csm_marker, score) if score else '%s%s' % (gene, csm_marker)
    ax.set_title(title, fontsize=fs_title, y=0.98, va='top')
    if show_ylabel:
        ax.set_ylabel('Strength')
    yt = np.concatenate((stain_gloms, ys_mean)) if stain_dict is not None else ys_mean
    xrange, yrange = xs[-1] - xs[0], np.nanmax(yt) - np.nanmin(yt)
    xlim, ylim = [-.008 * xrange, xs[-1] + .008 * xrange], [np.nanmin(yt) - .1 * yrange, np.nanmax(yt) + .1 * yrange]
    xticklabels_ = gloms if show_xticklabels else None
    ph.adjust_spines(ax, show_axes, lw=1, xlim=xlim, ylim=ylim, xticks=xs, xticklabels=xticklabels_, xticklabel_ro='vertical')

def plot_expr_afo_gloms_linearaxis_multigenes(seqss, genes, ncol=2, **kwargs):
    nrow = int((len(genes)-1)/ncol)+1
    col_size = 2.8 if ncol == 2 else 2
    plt.figure(1, (17, nrow*col_size))
    gs = gridspec.GridSpec(nrow, ncol)
    for i, gene in enumerate(genes):
        row = int(i/ncol)
        col = int(i%ncol)
        ax = plt.subplot(gs[row,col])
        show_xticklabels, show_ylabel = False, False
        if col == 0:
            if row == nrow-1:
                show_xticklabels, show_ylabel = True, True
        plot_expr_afo_gloms_linearaxis(seqss, gene, ax=ax, show_xticklabels=show_xticklabels, show_ylabel=show_ylabel,
                                       fs=9, **kwargs)

def plot_expr_afo_gloms_onmap_v1_bothLR(seqss, mm, gene='Ten-m', glom_list=[], stain_dict=None, datatype_orn=True, axs=[],
                              z_projection=True, fs=13, fs_title=16, show_title=True):
    # if glom_list is empty, plot all available gloms either from seq data, or from stain dict
    if not len(glom_list):
        if stain_dict is not None:
            stain_dict_sorted = dict(sorted(stain_dict.items(), reverse=True, key=lambda item: item[1]))
            glom_list = list(stain_dict_sorted.keys())
        else:
            glom_list = list(np.unique(get_glomtypes_from_seqss(seqss, datatype_orn)))
    if stain_dict is not None:
        glom_values = [stain_dict[glom] for glom in glom_list]
    else:
        seqss_1gene = seqss[:, (seqss.var_names == gene)]
        glom_values = []
        for glom_ in glom_list:
            seqss_1gene1glom = seqss_1gene[get_glomtypes_from_seqss(seqss, datatype_orn) == glom_]
            if len(seqss_1gene1glom):
                glom_values.append(np.nanmean(seqss_1gene1glom.X, axis=0))
            else:
                glom_values.append(0)

    # calculate Matrix
    # cmap = plt.cm.OrRd if datatype_orn else plt.cm.GnBu
    cmap = ph.DReds if datatype_orn else ph.GnBls
    axis_projection = 2 if z_projection else 0
    glom_xys = []
    for iglom, glom_ in enumerate(glom_list):
        glom_ids = get_glomid([glom_], infunc_print=False)
        for glom_id in glom_ids:
            if glom_id >= 0:
                glom_xyz = (mm == glom_id + 1)
                a_bool = np.ndarray.max(glom_xyz, axis=axis_projection)
                b_float = np.full_like(a_bool, np.nan, dtype=float)
                b_float[a_bool] = 1
                glom_xy = np.array(b_float * glom_values[iglom])
                glom_xys.append(glom_xy)

    glom_xys = np.array(glom_xys)
    M = np.nanmean(glom_xys, axis=0)
    if not z_projection:
        M = M.T[::-1,:]
    M[np.isnan(M)] = -.1
    cmax = np.ndarray.max(M)

    # plot
    if z_projection:
        _ = plt.figure(1, (5.7, 3))
    else:
        _ = plt.figure(1, (5.7, 1.5))
    if not len(axs):
        gs = gridspec.GridSpec(1, 50)
        axs = [plt.subplot(gs[0, :46]), plt.subplot(gs[0, 46])]
    ax = axs[0]
    ax.set_aspect('equal', 'datalim')
    xs = np.arange(0, M.shape[1] + 1)
    ys = np.arange(0, M.shape[0] + 1)
    ph.adjust_spines(ax, [], ylim=[0, M.shape[0]], xlim=[0, M.shape[1]], )
    _ = ax.pcolormesh(xs, ys, M, cmap=cmap)
    if show_title:
        csm_marker = '*' if gene in CSMs else ''
        title = '%s%s' % (gene, csm_marker)
        ax.set_title(title, fontsize=fs_title, loc='center', y=1, va='bottom')

    ax = axs[1]
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient)).T
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_yticklabels([0,'%1.1f' % cmax], fontsize=fs)
    ax.yaxis.tick_right()
    ax.tick_params('both', length=2, width=1, which='major')
    # ax.set_ylabel('$\Delta$F/F', rotation=0, va='bottom', ha='left')
    ax.yaxis.set_label_position("right")
    ph.adjust_spines(ax, ['right'], pad=0, lw=0, ylim=[0, 255], yticks=[0, 255], xticks=[])
    ax.spines['right'].set_visible(False)

def plot_expr_afo_gloms_onmap_ORNPN_v1_bothLR(seqsss,  mm, gene='Ten-m', glom_list=[], z_projection=True, **kwargs):
    if z_projection:
        _ = plt.figure(1, (11.2, 3.1))
    else:
        _ = plt.figure(1, (11.2, 1.4))
    gs = gridspec.GridSpec(3, 100, height_ratios=[1,30,1])
    seqss_ORN, seqss_PN = seqsss[0], seqsss[1]
    plot_expr_afo_gloms_onmap(seqss_ORN, mm, axs=[plt.subplot(gs[:, :47]), plt.subplot(gs[1, 48])], z_projection=z_projection,
                              gene=gene, glom_list=glom_list, datatype_orn=True, show_title=True, **kwargs)
    plot_expr_afo_gloms_onmap(seqss_PN, mm, axs=[plt.subplot(gs[:, 51:98]), plt.subplot(gs[1, 99])], z_projection=z_projection,
                              gene=gene, glom_list=glom_list, datatype_orn=False, show_title=False, **kwargs)

def plot_expr_afo_gloms_onmap(seqss, mm, gene='Ten-m', glom_list=[], stain_dict=None, datatype_orn=True, axs=[],
                              z_projection=True, fs=13, fs_title=16, show_title=True, show_cbar=True,
                              vs=[], title=None):
    # if glom_list is empty, plot all available gloms either from seq data, or from stain dict
    if not len(glom_list):
        if stain_dict is not None:
            stain_dict_sorted = dict(sorted(stain_dict.items(), reverse=True, key=lambda item: item[1]))
            glom_list = list(stain_dict_sorted.keys())
        else:
            glom_list = list(np.unique(get_glomtypes_from_seqss(seqss, datatype_orn)))
    if stain_dict is not None:
        glom_values = [stain_dict[glom] for glom in glom_list]
    else:
        seqss_1gene = seqss[:, (seqss.var_names == gene)]
        glom_values = []
        for glom_ in glom_list:
            seqss_1gene1glom = seqss_1gene[get_glomtypes_from_seqss(seqss, datatype_orn) == glom_]
            if len(seqss_1gene1glom):
                glom_values.append(np.nanmean(seqss_1gene1glom.X, axis=0))
            else:
                glom_values.append(-.1)

    # calculate Matrix
    mwidth = mm.shape[1]
    idx_ = int(mwidth/2)
    mm = mm[:,idx_:,:]
    cmap = ph.DReds if datatype_orn else ph.GnBls
    axis_projection = 2 if z_projection else 0
    glom_xys = []
    for iglom, glom_ in enumerate(glom_list):
        glom_ids = get_glomid([glom_], infunc_print=False)
        for glom_id in glom_ids:
            if glom_id >= 0:
                glom_xyz = (mm == glom_id + 1)
                a_bool = np.ndarray.max(glom_xyz, axis=axis_projection)
                b_float = np.full_like(a_bool, np.nan, dtype=float)
                b_float[a_bool] = 1
                glom_xy = np.array(b_float * glom_values[iglom])
                glom_xys.append(glom_xy)

    glom_xys = np.array(glom_xys)
    M = np.nanmean(glom_xys, axis=0)
    if not z_projection:
        M = M.T[::-1,:]
    M[np.isnan(M)] = -.1
    cmax = np.ndarray.max(M)

    # plot
    if not len(axs):
        if z_projection:
            _ = plt.figure(1, (5.7, 3))
        else:
            _ = plt.figure(1, (5.7, 1.5))
        gs = gridspec.GridSpec(1, 50)
        axs = [plt.subplot(gs[0, :46]), plt.subplot(gs[0, 46])] if show_cbar else [plt.subplot(gs[0, :46]),]
    ax = axs[0]
    ax.set_aspect('equal', 'datalim')
    xs = np.arange(0, M.shape[1] + 1)
    ys = np.arange(0, M.shape[0] + 1)
    ph.adjust_spines(ax, [], ylim=[0, M.shape[0]], xlim=[0, M.shape[1]], )
    if len(vs):
        _ = ax.pcolormesh(xs, ys, M, cmap=cmap, vmin=vs[0], vmax=vs[1])
    else:
        _ = ax.pcolormesh(xs, ys, M, cmap=cmap)
    if show_title:
        if title is None:
            csm_marker = '*' if gene in CSMs else ''
            title = '%s%s' % (gene, csm_marker)
        ax.set_title(title, fontsize=fs_title, loc='center', y=1, va='bottom')

    if show_cbar:
        ax = axs[1]
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient)).T
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        if len(vs):
            ax.set_yticklabels(['%2.1f' % vs[0], '%2.1f' % vs[1]], fontsize=fs)
        else:
            ax.set_yticklabels([0,'%2.1f' % cmax], fontsize=fs)
        ax.yaxis.tick_left()
        ax.tick_params('both', length=2, width=1, which='major')
        # ax.set_ylabel('$\Delta$F/F', rotation=0, va='bottom', ha='left')
        ax.yaxis.set_label_position("left")
        ph.adjust_spines(ax, ['left'], pad=0, lw=0, ylim=[0, 255], yticks=[0, 255], xticks=[])
        ax.spines['left'].set_visible(False)

def plot_expr_afo_gloms_onmap_ORNPN(seqsss, mm, genes=['Ten-m'], datatypes_orn=[], glom_list=[], show_title=True,
                                    vss=[], **kwargs):
    ngene = len(genes)
    _ = plt.figure(1, (3.3*ngene, 4.5))
    gs = gridspec.GridSpec(6, 50*ngene, height_ratios=[1,30,1,1,10.5,1])
    for i in range(ngene):
        seqss = seqsss[i] if type(seqsss) is list else seqsss
        datatype_orn = datatypes_orn[i] if type(datatypes_orn) is list else datatypes_orn
        glom_list_ = glom_list if type(glom_list) is list else glom_list[i]  # could use tuple
        show_title_ = show_title
        vs = [] if not len(vss) else vss[i]
        # plot z projection
        axs = [plt.subplot(gs[:3, i*50:(i*50+45)]), plt.subplot(gs[1, (i*50+45):(i*50+47)])]
        plot_expr_afo_gloms_onmap(seqss, mm, axs=axs, z_projection=True, gene=genes[i], glom_list=glom_list_,
                                  datatype_orn=datatype_orn, show_title=show_title_, show_cbar=True, vs=vs, **kwargs)
        # plot y projection
        axs = [plt.subplot(gs[3:, i*50:(i*50+45)]), ]
        plot_expr_afo_gloms_onmap(seqss, mm, axs=axs, z_projection=False, gene=genes[i], glom_list=glom_list_,
                                  datatype_orn=datatype_orn, show_title=False, show_cbar=False, vs=vs, **kwargs)

def plot_expr_afo_gloms_onmap_ORNPN_matching(seqsss, mm, genes=['Ten-m'], datatypes_orn=[], glom_list=[], titles=[], **kwargs):
    ngene = len(genes)
    _ = plt.figure(1, (3.3*ngene, 4.5))
    gs = gridspec.GridSpec(6, 50*ngene, height_ratios=[1,30,1,1,10.5,1])
    for i in range(ngene):
        seqss = seqsss[i] if type(seqsss) is list else seqsss
        datatype_orn = datatypes_orn[i] if type(datatypes_orn) is list else datatypes_orn
        glom_list_ = glom_list if type(glom_list) is list else glom_list[i]  # could use tuple
        title_ = genes[0] if i == 0 else '%s, slope=%2.2f, r=%2.2f' % (genes[1], titles[0], titles[1])
        # plot z projection
        axs = [plt.subplot(gs[:3, i*50:(i*50+45)]), plt.subplot(gs[1, (i*50+45):(i*50+47)])]
        plot_expr_afo_gloms_onmap(seqss, mm, axs=axs, z_projection=True, gene=genes[i], glom_list=glom_list_,
                                  datatype_orn=datatype_orn, show_title=True, show_cbar=True, title=title_, **kwargs)
        # plot y projection
        axs = [plt.subplot(gs[3:, i*50:(i*50+45)]), ]
        plot_expr_afo_gloms_onmap(seqss, mm, axs=axs, z_projection=False, gene=genes[i], glom_list=glom_list_,
                                  datatype_orn=datatype_orn, show_title=False, show_cbar=False, **kwargs)















# staining dictionary
CSMs = ['18w','2mit','5-HT2B','7B2','Acer','Acp62F','Actbeta','AdamTS-A','Adgf-A','Adgf-A2','Adgf-B','Adgf-C','Adgf-D',
        'Adgf-E','AdoR','Ag5r','Ag5r2','Akh','Alk','alrm','Ama','amn','amon','ana','Ance','Ance-2','Ance-3','Ance-4',
        'Ance-5','Ank2','antr','aos','Rfabg','Apoltp','Appl','arr','AstA','AstA-R1','AstA-R2','AstC','AstC-R1','AstC-R2',
        'atk','axo','b6','babo','babos','bark','bdl','beat-Ia','beat-Ib','beat-Ic','beat-IIa','beat-IIb','beat-IIIa',
        'beat-IIIb','beat-IIIc','beat-IV','beat-Vb','beat-Vc','beat-VI','beat-VII','beta3GalTII','bnl','boi','boly',
        'boss','botv','brn','Bsg','bt','btl','Btnd','C1GalTA','C901','Cad74A','Cad86C','Cad87A','Cad88C','Cad89D',
        'Cad96Ca','Cad96Cb','Cad99C','CadN','CadN2','Cals','Capa','CapaR','caps','CCAP','CCAP-R','CCHa1-R','CCHa2-R',
        'CCKLR-17D1','CCKLR-17D3','Ccn','Cda4','Cda5','Cda9','cDIP','Cep97','CG10005','CG10182','CG10183','CG10307',
        'CG10345','CG10359','CG10481','CG10483','CG10650','CG10651','CG10663','CG10702','CG10725','CG10731','CG1077',
        'CG11099','CG11318','CG11353','CG11357','CG11374','CG11377','CG11425','CG11426','CG11437','CG11438','CG11635',
        'CG11807','CG11977','CG12004','CG12009','CG12290','CG12484','CG12594','CG12716','CG12746','CG12860','CG12861',
        'CG12913','CG12950','CG12990','CG13101','CG13196','CG13229','CG13252','CG13310','CG13312','CG13313','CG13325',
        'CG13492','CG13506','CG13532','CG13575','CG13579','CG13614','CG13616','CG13643','CG13676','CG13708','CG13865',
        'CG13869','CG13937','CG13950','CG13992','CG13995','CG14024','CG14115','CG14141','CG14185','CG14204','CG14205',
        'CG14219','CG14280','CG14325','CG14372','CG14662','CG14720','CG14760','CG14762','CG14826','CG14829','CG14834',
        'CG14879','CG14880','CG14964','CG14995','CG15020','CG1504','CG15153','CG15170','CG15312','CG15347','CG15533',
        'CG15534','CG15556','CG15630','CG15721','CG15744','CG15765','CG15773','CG1632','CG16723','CG16868','CG16959',
        'CG16974','CG17047','CG17111','CG17211','CG17264','CG17575','fas','CG17739','CG17780','CG17784','CG17826',
        'CG17839','CG17906','CG1791','CG17974','CG18095','CG18208','CG18249','CG18480','CG18558','CG18870','CG1889',
        'CG2061','CG2127','CG2247','CG2736','CG2839','CG2901','CG2975','CG2983','CG2991','CG30036','CG30037','CG30046',
        'CG30197','CG30203','CG30371','CG30375','CG3038','CG3040','CG30471','CG30486','CG31051','CG3106','CG31075',
        'CG31076','CG31077','CG3119','CG31221','CG31296','CG31431','CG31635','CG31637','CG31741','CG31743','CG31814',
        'CG31869','CG31999','CG32055','CG32085','CG32206','CG32354','CG32373','CG32432','CG32547','CG32645','CG32647',
        'CG32679','CG32687','CG3277','CG33143','CG33337','CG33339','CG33340','CG33543','CG33639','CG3376','CG33958',
        'CG33985','CG33986','CG34056','CG34057','CG3408','CG34113','CG34114','CG34353','CG34355','CG34357','CG34370',
        'CG34371','CG34402','CG34428','CG34429','CG3494','CG3640','CG3829','CG4096','CG4168','CG42255','CG42296',
        'CG42302','CG42313','CG42326','CG42329','CG42346','CG42369','CG42370','CG42389','CG4250','CG42537','CG42564',
        'CG42613','CG42709','CG42728','CG42729','CG42811','CG42812','CG42830','CG4313','CG43313','CG43332','CG43333',
        'CG4363','CG4377','CG43775','CG43776','CG43777','CG43778','CG43779','CG43896','CG44153','CG4480','CG45263',
        'CG4681','CG4691','CG4734','CG4781','CG4793','CG4835','CG4950','CG4998','CG5096','CG5335','CG5510','CG5597',
        'CG5639','CG5756','CG5758','CG5768','CG5810','CG5819','CG5850','CG5888','CG5892','CG6024','CG6495','CG6553',
        'CG6628','CG6739','CG6749','CG6788','CG6867','CG6959','CG7166','CG7173','CG7179','CG7201','CG7227','CG7248',
        'CG7252','CG7298','CG7352','CG7381','CG7457','CG7466','CG7497','CG7509','CG7536','CG7565','CG7607','CG7668',
        'CG7702','CG7800','CG7896','CG8072','CG8112','CG8170','CG8180','CG8184','CG8197','CG8213','CG8272','CG8399',
        'CG8483','CG8642','CG8668','CG8673','CG8701','CG8708','CG8852','CG8909','CG9044','CG9095','CG9109','CG9220',
        'CG9289','CG9400','CG9550','CG9555','CG9572','CG9593','CG9643','CG9822','ChLD3','chp','Cht3','Cht5','Cht6',
        'Cht7','Cirl','CNMaR','comm','comm2','Con','Cont','conv','Corin','Cow','CR10032','crb','crq','Crz','cue',
        'CG17352','cv','cv-2','cv-d','cyr','dally','daw','Ddr','Dg','Dh31','Dh31-R','Dh44','Dh44-R1','Dh44-R2',
        'DIP-alpha','DIP-beta','DIP-delta','DIP-epsilon','DIP-eta','DIP-gamma','DIP-iota','DIP-lambda','DIP-theta',
        'DIP-zeta','Dl','dlp','dnt','dome','Dop2R','DopEcR','dpp','dpr1','dpr10','dpr11','dpr12','dpr13','dpr14',
        'dpr15','dpr16','dpr17','dpr18','dpr19','dpr2','dpr20','dpr21','dpr3','dpr4','dpr5','dpr6','dpr7','dpr8',
        'dpr9','dpy','drl','Drl-2','drpr','ds','dsb','Dscam1','Dscam2','Dscam3','Dscam4','dsd','Dsk','dy','dyl','eater',
        'ed','Edem2','Egfr','egh','egr','Eh','emp','Eph','ergic53','ETH','ETHR','Ext2','eys','f-cup','Fas1','Fas2',
        'Fas3','fat-spondin','fend','Fili','fliI','FMRFa','FMRFaR','fng','fog','fra','frac','fred','frj','Fs','ft',
        'Fur1','Fur2','fw','fz','fz2','fz3','fz4','galectin','GalT1','Gasp','gbb','GCS2beta','gd','Gel','GLaz','glec',
        'Gli','Glt','GNBP1','GNBP2','GNBP3','gogo','Gp150','Gpb5','grk','haf','Hasp','hbs','hec','hfw','hig','Hml',
        'Hrd3','Hs2st','Hs3st-A','Hs3st-B','Hs6st','htl','hubl','Hug','ics','if','ihog','Ilp1','Ilp2','Ilp3','Ilp4',
        'Ilp6','Ilp7','ImpE1','ImpL1','ImpL2','InR','IntS14','IntS6','ItgaPS4','ItgaPS5','Itgbn','ITP','jeb','Kal1',
        'kek1','kek2','kek3','kek4','kek5','kek6','kirre','klg','knk','kon','Krn','kug','Kul','kuz','l(1)G0289','Lac',
        'LanA','LanB1','LanB2','Lap1','Lapsyn','Lar','laza','lbk','lbm','lectin-22C','lectin-37Da','lectin-37Db','Lerp',
        'Lgr1','Lgr3','Lgr4','Lk','Lkr','loh','lox','lox2','LPCAT','LpR1','LpR2','Lrch','LRP1','LRR','Lrr47','Lrrk',
        'Lrt','ltl','m','CG12796 ','magu','Manf','Mapmodulin','mav','mbc','Mco1','Mcr','mdy','Meltrin','mesh','mew',
        'mey','mfas','mgl','Mip','miple1','miple2','mlt','mmd','Mmp1','Mmp2','modSP','moody','Mp','Ms','mspo','MsR1',
        'MsR2','MstProx','mtd','mth','mthl1','mthl10','mthl11','mthl2','mthl3','mthl4','mthl5','mthl6','mthl7','mthl8',
        'mthl9','Mtp','Muc11A','Muc68Ca','Muc68D','Muc68E','Mur29B','Mur2B','Mur89F','mys','N','nahoda','Ndg','ndl',
        'neo','nes','NetA','NetB','Neto','NimA','NimB2','NimB4','NimC1','NimC2','NimC4','ninaD','NLaz','Nlg1','Nlg2',
        'Nlg3','Nlg4','nolo','nompA','nord','Npc2a','NPF','Nplp1','Nplp2','Nplp3','Nplp4','Nrg','Nrk','nrm','Nrt',
        'Nrx-1','Nrx-IV','NT1','nyo','obst-B','obst-H','Octbeta1R','Octbeta2R','Octbeta3R','otk','otk2','oxt','oys',
        'Pdf','Pdfr','pes','Pex23','Phlpp','pio','pip','PK1-R','PK2-R1','PK2-R2','PlexA','PlexB','plum','por','pot',
        'ppk12','Ppn','Proc','Proc-R','Ptp10D','Ptp4E','Ptp52F','Ptp69D','Ptp99A','put','Pvf1','Pvf2','Pvf3','Pvr',
        'pwn','Pxn','pyr','qsm','RanGAP','rasp','Rcd2','rdo','Reck','Ret','rgn','rk','robo1','robo2','robo3','Ror',
        'rost','Rpn13','Rpn13R','rst','RYa-R','santa-maria','Sap-r','sas','sax','Sb','sbr','sca','scb','Scgalpha',
        'Sclp','scpr-A','scpr-B','scpr-C','scramb1','scw','Sdc','sdk','Sdr','sds22','Sema-1a','Sema-1b','Sema-2a',
        'Sema-2b','Sema-5c','Ser','serp','sev','shf','shg','side','SIFaR','Skeletor','sli','slow','CG34380','smo',
        'Snmp1','Snmp2','sNPF','sNPF-R','sns','sog','SP2353','SPARC','spi','spz','spz3','spz4','spz5','spz6','Sr-CI',
        'Sr-CII','Sr-CIII','Sr-CIV','srw','ssp7','stan','Strn-Mlck','Sur-8','swi2','swif','Swim','Tace','TbCMF46',
        'Tehao','Ten-a','Ten-m','Tep1','Tep2','Tep3','Tep4','Tep5','Tequila','tgy','ths','CG14608','Tig','Tk','TkR86C',
        'TkR99D','tkv','Tl','tld','tnc','tok','Toll-4','Toll-6','Toll-7','Toll-9','Tollo','tor','TrissinR','trk','trn',
        'trol','tsg','Tsp','Tsp26A','Tsp29Fa','Tsp29Fb','Tsp2A','Tsp33B','Tsp39D','Tsp3A','Tsp42A','Tsp42Ea','Tsp42Eb',
        'Tsp42Ec','Tsp42Ed','Tsp42Ee','Tsp42Ef','Tsp42Eg','Tsp42Eh','Tsp42Ei','Tsp42Ej','Tsp42Ek','Tsp42El','Tsp42En',
        'Tsp42Eo','Tsp42Ep','Tsp42Eq','Tsp42Er','Tsp47F','Tsp5D','Tsp66A','Tsp66E','Tsp68C','Tsp74F','Tsp86D','Tsp96F',
        'Tsp97E','ttv','tutl','twin','tyn','TyrR','TyrRII','uif','unc-5','Unc-89','upd1','Vang','verm','vn','wb','wdp',
        'wg','wgn','wit','Wnt10','Wnt2','Wnt4','Wnt5','Wnt6','wntD','wrapper','wry','Wsck','wun','wun2','yl','zormin',
        'zye',]
glomid_list = ['VA1d_R', 'VA1d_L', 'VA1v_R', 'VA1v_L', 'DA1_R', 'DA1_L', 'DA3_R', 'DA3_L', 'DL3_R', 'DL3_L',
          'VA6_R', 'VA6_L', 'DA4l_R', 'DA4l_L', 'DA4m_R', 'DA4m_L', 'D_R', 'D_L', 'DA2_R', 'DA2_L',
          'DC3_R', 'DC3_L', 'DC1_R', 'DC1_L', 'DC4_R', 'DC4_L', 'DC2_R', 'DC2_L', 'DM3_R', 'DM3_L',
          'DL5_R', 'DL5_L', 'DL1_R', 'DL1_L', 'DL4_R', 'DL4_L', 'DP1m_R', 'DP1m_L', 'DP1l_R', 'DP1l_L',
          'DL2l_R', 'DL2l_L', 'DL2d_R', 'DL2d_L', 'VL2a_R', 'VL2a_L', 'VL2p_R', 'VL2p_L', 'VL1_R', 'VL1_L',
          'V_R', 'V_L', 'VA5_R', 'VA5_L', 'VA3_R', 'VA3_L', 'VA4_R', 'VA4_L', 'VA2_R', 'VA2_L',
          'VM2_R', 'VM2_L', 'VM3_R', 'VM3_L', 'VM1_R', 'VM1_L', 'VM6_R', 'VM6_L', 'VM4_R', 'VM4_L',
          'VC3_R', 'VA7l_R', 'VA7l_L', 'VA7m_R', 'VA7m_L', 'VC2_R', 'VC2_L', 'VC1_R', 'VC1_L', 'VM5v_R',
          'VM5v_L', 'VM5d_R', 'VM5d_L', 'DM6_R', 'DM6_L', 'DM5_R', 'DM5_L', 'DM2_R', 'DM2_L', 'VC3_L',
          'DM1_R', 'DM1_L', 'DM4_R', 'DM4_L', 'VM7v_R', 'VM7v_L', 'VM7_R', 'VM7_L', 'VC4_R', 'VC4_L']
glomor = {'DA4m':'Or2a', 'DL5':'Or7a', 'VM3':'Or9a', 'DL1':'Or10a', 'DC2':'Or13a', 'DC1':'Or19a_Or19b',
          'DM2':'Or22a_Or22b', 'DA3':'Or23a', 'VC1':'Or33c', 'VC3':'Or35a', 'VM7':'Or42a', 'DM1':'Or42b',
          'DA4l':'Or43a', 'VM2':'Or43b', 'VA7l':'Or46a', 'DM3':'Or47a', 'VA1v':'Or47b', 'DL4':'Or49a', 'VA5':'Or49b',
          'DA2':'Or56a', 'DM4':'Or59b', 'VM7':'Or59c', 'DL3':'Or65a_Or65b_Or65c', 'DM6':'Or67a', 'VA3':'Or67b',
          'DA1':'Or67d', 'VA6':'Or67d', 'D':'Or69a', 'VC2':'Or71a', 'VA6':'Or82a', 'DC3':'Or83c','DM5':'Or85a',
          'VA4':'Or85d', 'VC1':'Or85e', 'DL4':'Or85f', 'VA1d':'Or88a', 'VA2':'Or92a', 'VM5v':'Or98a', 'V':'Gr21a',}
all_off = {'DA1':0, 'VA1d':0, 'VA1v':0, 'DL3':0, 'DA3':0, 'D':0, 'DA4m':0, 'DA4l':0, 'DA2':0, 'VA6':0, 'VA7m':0,
           'VA7l':0, 'VA5':0, 'DM6':0, 'VM5d':0, 'VM5v':0, 'VA2':0, 'VA3':0, 'DL4':0, 'DC3':0, 'DC1':0, 'DC2':0,
           'DM2':0, 'DM5':0, 'VM7v':0, 'VM2':0, 'VM3':0, 'VC2':0, 'VA4':0, 'DL1':0, 'DL5':0, 'DM3':0, 'DC4':0, 'DL2':0,
           'VM7':0, 'VC4':0, 'VC1':0, 'VC3':0, 'VM4':0, 'VL2a':0, 'DL2':0, 'DP1l':0, 'DP1m':0, 'DM1':0, 'DM4':0,
           'VL2p':0, 'VL1':0, 'V':0, 'VM6':0, 'VM1':0, }
fili_stain_ORN = {'D': 1, 'DL5':1, 'DL1':1, 'DL3':0, 'DP1m':1, 'DL4':1, 'DA3': 1, 'DA4l':1, 'DC3':0, 'DA1':0,
                 'VA1d':1, 'VA1v':1, 'DL2':1, 'VL2p':0, 'VL2a':1, 'VL1':0, 'VA5':1, 'VA7l':1, 'VC2':1, 'VA6':1,
                 'VA4':0, 'V':1, 'VM6':1, 'VA3':1, 'VA2':1, 'VM1':0, 'VA7m':0, 'VM5v':0, 'VM2':1, 'DM5':1, 'VM5d':0,
                 'DM4':0, 'DM2':1, 'DC4':1, 'DC0':0, 'DA2':0, 'DA4m':0, 'DM3':0, 'DM1':1, 'DM6':0}
fili_stain_PN = {'D': 0, 'DL5':0, 'DL1':0, 'DL3':0, 'DP1m':1, 'DL4':1, 'DA4l':0, 'DC3':1, 'DA1':1,
                 'VA1d':1, 'VA1v':0, 'DL2':1, 'VL2p':0, 'VL2a':0, 'VL1':0, 'VA5':1, 'VA7l':0, 'VC2':0, 'VA6':0,
                 'VA4':1, 'VC1':0, 'VA3':0, 'VA2':1, 'VM1':0, 'VA7m':0, 'VM2':1, 'DM5':1,
                 'DM4':1, 'DM2':1, 'DC4':0, 'DC2':0, 'DA2':0, 'DA4m':0, 'DM3':1, 'DM1':1, 'DM6':0}
tenm_stain_ORN = {'DA3':1, 'VA1d':1, 'VA1v':1, 'VA6':1, 'VM5d':1, 'VM5v':1, 'VA5':1, 'VA3':1, 'DL3':1, 'D':1, 'DA4m':1,
                 'DM2':1, 'DM5':1, 'DL1':1, 'DC4':1, 'V':1, 'DP1m':1, 'DM1':1, 'DM4':1,
                 'DA1':0, 'DC3':0, 'VA7l':0, 'VA4':0, 'VA2':0, 'DM3':0, 'DL5':0, 'DL2':0, 'VA1l':0, 'DP1l':0,}
tenm_stain_PN = {'DA3':1, 'VA1d':1, 'VA1v':1, 'VA6':1, 'VM5d':1, 'VM5v':1, 'VA5':1, 'VA3':1, 'DL3':1, 'D':1, 'DA4m':1,
                 'DM2':1, 'DM5':1, 'DL1':1, 'DC4':1, 'V':1, 'DP1m':1, 'DM1':1, 'DM4':1,
                 'DA1':0, 'DC3':0, 'VA7l':0, 'VA4':0, 'VA2':0, 'DM3':0, 'DL5':0, 'DL2':0, 'VA1l':0, 'DP1l':0,}
tena_stain_ORN = {'DA3':1, 'DA1':1, 'DL3':1, 'DC3':.5, 'DA2':.5, 'DC1':.5, 'D':.5, 'DM6':.5, 'DL5':.5, 'DM3':.5,
                  'VA1d':0, 'VA1v':0, 'DA4m':0, 'DA4l':0, 'VA6':0, 'VM5d':0, 'VM5v':0, 'VA7m':0, 'VA7l':0, 'VA5':0,
                  'VA3':0, 'VA2':0, 'DC2':0, 'DM2':0, 'DL4':0, 'DM5':0, 'VM2':0, 'VM3':0, 'VC2':0, 'VA4':0,
                  'DC4':0, 'DL1':0, 'DL2':0, 'VL2a':0, 'VC1':0, 'VM7':0, 'VC4':0, 'VC3':0, 'VM4':0,
                  'DM1':0, 'DM4':0, 'DP1m':0, 'DP1l':0, 'VL2p':0, 'VL1':0, 'V':0, 'VM1':0, 'VM6':0, }
tena_stain_PN = {'DA3':1, 'DA1':1, 'DL3':1, 'DC3':.5, 'DA2':.5, 'DC1':.5, 'D':.5, 'DM6':.5, 'DL5':.5, 'DM3':.5,
                  'VA1d':0, 'VA1v':0, 'DA4m':0, 'DA4l':0, 'VA6':0, 'VM5d':0, 'VM5v':0, 'VA7m':0, 'VA7l':0, 'VA5':0,
                  'VA3':0, 'VA2':0, 'DC2':0, 'DM2':0, 'DL4':0, 'DM5':0, 'VM2':0, 'VM3':0, 'VC2':0, 'VA4':0,
                  'DC4':0, 'DL1':0, 'DL2':0, 'VL2a':0, 'VC1':0, 'VM7':0, 'VC4':0, 'VC3':0, 'VM4':0,
                  'DM1':0, 'DM4':0, 'DP1m':0, 'DP1l':0, 'VL2p':0, 'VL1':0, 'V':0, 'VM1':0, 'VM6':0, }
caps_stain_PN = {'DM6':1, 'DM5':1, 'VA2':1, 'VA6':1, 'VA7m':1, 'VA5':1, 'VA1v':1, 'DM1':1, 'VM3':1, 'VA4':1, 'VC2':1,
                'DL4':1, 'DL2v':1, 'DL2d':1, 'VL2a':1, 'DM4':1, 'DM1':1, 'DM3':1, 'DL5':1, 'VL2p':1, 'VC1':1, 'VC3l':1,
                'VM1':1, 'DA2':0, 'DA4':0, 'DL3':0, 'DA1':0, 'VA1d':0, 'VA7l':0, 'VA3':0, 'VM2':0, 'D':0, 'DC1':0,
                'DC2':0, 'VM5':0, 'DC3':0, 'VM7':0, 'DP1m':0, 'DL1':0, 'VL1':0, 'V':0, }
caps_stain_ORN = {'DM6':1, 'DM5':1, 'VM2':1, 'VA2':1, 'VA6':1, 'VA7m':1, 'VA7l':1, 'VA3':1, 'VA1v':1,'DA1':1, 'DM2':1,
                  'VM3':1, 'VA4':1, 'DC2':1, 'DC3':1, 'VL2a':1, 'DM4':1, 'DM1':1, 'DM3':1, 'VM7':1, 'DP1m':1, 'DP1l':1,
                  'VC3m':1, 'VC3l':1, 'VL2p':1, 'VL1':1, 'VM4':1, 'VM6':1, 'DA2':0, 'DA3':0, 'DL3':0, 'DA4':0, 'VA1d':0,
                  'VA5':0, 'VM5':0, 'DC1':0, 'D':0, 'DL4':0, 'DL2d':0, 'DL2v':0, 'VC2':0, 'DL5':0, 'DL1':0, 'VC1':0,
                  'VM1':0, 'V':0,}
sema2b_stain_ORN = {'D':0, 'DL3':0, 'DA3':0, 'DA4m':0, 'DA4l':0, 'DA1':0, 'VA1d':0, 'VA1v':0, 'DC1':0, 'DC3':0,
                    'DL4':0, 'DL5':0, 'DL1':0, 'DL2':0, 'VL2a':0, 'VL2p':0, 'DP1l':0, 'DP1m':0,
                    'DM6':1, 'VM5d':1, 'VM5v':1, 'VA7m':1, 'VA7l':1, 'VA2':1, 'VA3':1, 'VA5':1, 'DM2':1, 'DM5':1,
                    'VM2':1, 'VM3':1, 'VA4':1, 'VM7':1, 'VC4':1, 'VC3':1, 'VM4':1, 'DM4':1, 'VM1':1, 'VM6':1, }
toll7_stain_ORN = {'DA1':1, 'VA1d':1, 'VA1v':1, 'DL3':0, 'DA3':0, 'D':0, 'DA4m':0, 'DA4l':0, 'DA2':0, 'VA6':0, 'VA7m':0,
           'VA7l':0, 'VA5':0, 'DM6':0, 'VM5d':0, 'VM5v':0, 'VA2':0, 'VA3':0, 'DL4':0, 'DC3':0, 'DC1':0, 'DC2':0,
           'DM2':0, 'DM5':0, 'VM7v':0, 'VM2':0, 'VM3':0, 'VC2':0, 'VA4':0, 'DL1':0, 'DL5':0, 'DM3':0, 'DC4':0, 'DL2':0,
           'VM7':0, 'VC4':0, 'VC1':0, 'VC3':0, 'VM4':0, 'VL2a':0, 'DL2':0, 'DP1l':0, 'DP1m':0, 'DM1':0, 'DM4':0,
           'VL2p':0, 'VL1':0, 'V':0, 'VM6':0, 'VM1':0, }
toll7_stain_PN = {'DA1':0, 'VA1d':0, 'VA1v':0,}
toll6_stain_PN = {'DA1':1, 'VA1d':1, 'VA1v':1, 'DL3':1, 'DC3':1, 'DC1':1, 'DA4l':1, 'DA4m':1,
                  'DA3': 0, 'D': 0, 'DA2': 0, 'VA6': 0, 'VA7m': 0, 'VA7l': 0, 'VA5': 0, 'DM6': 0, 'VM5d': 0, 'VM5v': 0,
                  'VA2': 0, 'VA3': 0, 'DL4': 0, 'DC2': 0, 'DM2': 0, 'DM5': 0, 'VM7v': 0, 'VM2': 0, 'VM3': 0, 'VC2': 0,
                  'VA4': 0, 'DL1': 0, 'DL5': 0, 'DM3': 0, 'DC4': 0, 'DL2': 0, 'VM7': 0, 'VC4': 0, 'VC1': 0, 'VC3': 0,
                  'VM4': 0, 'VL2a': 0, 'DL2': 0, 'DP1l': 0, 'DP1m': 0, 'DM1': 0, 'DM4': 0,
                  'VL2p': 0, 'VL1': 0, 'V': 0, 'VM6': 0, 'VM1': 0,}
toll6_stain_ORN = {'DA1':1, 'VA1d':1, 'VA1v':1, 'DL3':1, 'DC3':1, 'DC1':1, 'DA4l':1, 'DA4m':1,
                   'DA3': 0, 'D': 0, 'DA2': 0, 'VA6': 0, 'VA7m': 0, 'VA7l': 0, 'VA5': 0, 'DM6': 0, 'VM5d': 0, 'VM5v': 0,
                   'VA2': 0, 'VA3': 0, 'DL4': 0, 'DC2': 0, 'DM2': 0, 'DM5': 0, 'VM7v': 0, 'VM2': 0, 'VM3': 0, 'VC2': 0,
                   'VA4': 0, 'DL1': 0, 'DL5': 0, 'DM3': 0, 'DC4': 0, 'DL2': 0, 'VM7': 0, 'VC4': 0, 'VC1': 0, 'VC3': 0,
                   'VM4': 0, 'VL2a': 0, 'DL2': 0, 'DP1l': 0, 'DP1m': 0, 'DM1': 0, 'DM4': 0,
                   'VL2p': 0, 'VL1': 0, 'V': 0, 'VM6': 0, 'VM1': 0,}





















































