import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
import os, random, math
from scipy import stats
import functions_py37 as fc

redorange = (.96, .59, .2)
ddorange = (102/255., 51/255., 0/255.)
dorange = (0.85, 0.54, 0.24)
purple = (0.5, 0.2, .6)
dpurple = (0.2, 0.2, .6)
nblue = (.11, .27, .49)
bblue = (0.51, 0.67, 0.89)
ngreen = (0.01, 0.3, 0.01)
# green = (0.5, 0.75, 0.42)
green = (0, 0.7, 0)
dgreen = (0.3, 0.55, 0.22)
ddgreen = (0.1, 0.35, 0.02)
bgreen = (.34, .63, .56)
dred = (.52, .12, 0)
dred_pure = (.4, 0,0)
dblue_pure = (0,0,.5)
blue_green = (0,.3,.8)
blue_red = (0,0,1)
green_red = (.7,.7,0)


yellow = (.95, .74, .22)
nyellow = (.3, .3, 0)

magenta = (175/255.,101/255.,168/255.)
orange = (244/255., 135/255., 37/255.)
dred = (131/255., 35/255., 25/255.)
red = (204/255., 0/255., 0/255.)
blue = (37/255.,145/255.,207/255.)
dblue = (27/255.,117/255.,187/255.)
rose = (194/255.,52/255.,113/255.)
pink = (221/255., 35/255., 226/255.)
dimred = (221/255., 49/255., 49/255.)
brown = (71/255., 25/255., 2/255.)


black = (0., 0., 0.)
grey9 = (.1, .1, .1)
grey8 = (.2, .2, .2)
grey7 = (.3, .3, .3)
grey6 = (.4, .4, .4)
grey5 = (.5, .5, .5)
grey4 = (.6, .6, .6)
grey3 = (.7, .7, .7)
grey2 = (.8, .8, .8)
grey15 = (.85, .85, .85)
grey1 = (.9, .9, .9)
grey05 = (.95, .95, .95)
grey03 = (.97, .97, .97)
white = (1., 1., 1.)
lgrey = grey3
grey = grey5
dgrey = grey7

def create_linear_cm(RGB=[244,118,33]):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(RGB[0]/255., 1, N)[::-1]
    vals[:, 1] = np.linspace(RGB[1]/255., 1, N)[::-1]
    vals[:, 2] = np.linspace(RGB[2]/255., 1, N)[::-1]
    vals[0,:] = 0
    newcmp = ListedColormap(vals)
    return newcmp



Reds = create_linear_cm(RGB=[204,0,0])
Oranges = create_linear_cm(RGB=[204,102,0])
Yellows = create_linear_cm(RGB=[180,180,0])
GrassGreens = create_linear_cm(RGB=[102,204,0])
Greens = create_linear_cm(RGB=[0,152,0])
Cyans = create_linear_cm(RGB=[0,153,153])
CyanBlues = create_linear_cm(RGB=[0,102,204])
Blues = create_linear_cm(RGB=[0,0,204])
Purples = create_linear_cm(RGB=[102,0,204])
Pinks = create_linear_cm(RGB=[204,0,204])
Magentas = create_linear_cm(RGB=[204,0,102])
GnBls = create_linear_cm(RGB=[0,76,76])
DryReds = create_linear_cm(RGB=[153,76,52])
DReds = create_linear_cm(RGB=[152,0,0])



def colors_light():

    return [(1,.4,.4), (.4,.7,1), (1,1,.4), (1,.4,1), (.4,1,.7),
            (.4,1,1), (1,.7,.4), (.7,.4,1), (.7,1,.4), (1,.4,.7)]

def colors_dark():
    return [(.8,0,0), (0,.4,.8), (.6,.6,0), (.8,0,.8), (0,.8,.4),
            (.4,.8,0), (0,.8,.8), (.4,0,.8), (.8,0,.4), (.8,.4,0), ]

def colors_high_contrast():
    return [[.2,.2,.2],[.8,0,0],[.8,.6,0], [.6,.8,0,],[0,.8,.6],
            [0,.4,.8],[.5,0,1],[.8,0,.8],
            [.4,0,.4],
            [.4,.4,0],[0,.4,.2],[0,0,.4],[.4,0,.2],
            [1,0,0],[.5,1,0],[0,1,1],[.5,0,1],
            [1,.5,0],[0,1,0],[0,.5,1],[1,0,1],
            # [1,1,0],[0,1,.5],[0,0,1],[1,0,.5],
            [0, 1, .5], [0, 0, 1], [1, 0, .5],
            ]

def set_fontsize(fontsize=20):
    rcParams['font.size'] = fontsize
    rcParams['xtick.labelsize'] = fontsize
    rcParams['ytick.labelsize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    # rcParams["font.family"] = "Helvetica Neue"
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']

def adjust_spines(ax, spines, lw=.5, xlim=None, ylim=None,
                  xticks='auto', xticks_minor='none', yticks='auto', yticks_minor='none', xticklabel_ro='horizontal',
                  xticklabels=[], yticklabels=[], color='black', pad=1, ticklength=2, ticklw=1.5):
    if not lw is 'none':
        rcParams['axes.linewidth'] = lw
        ax.tick_params('both', length=ticklength, width=lw, which='major')
        ax.tick_params('both', length=ticklength * .66, width=lw, which='minor')

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', pad))  # outward by 10 points
            # spine.set_smart_bounds(True)
            spine.set_color(color)
        else:
            spine.set_color('none')  # don't draw spine

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    if 'bottom' in spines or 'top' in spines:
        if type(xticks) is str:
            if xticks == 'none':
                ax.set_xticks([])
        else:
            ax.set_xticks(xticks)
        if type(xticks_minor) is str:
            if xticks_minor == 'none':
                ax.set_xticks([], minor=True)
        else:
            ax.set_xticks(xticks_minor, minor=True)
        if xticklabels is None:
            ax.set_xticklabels([])
        elif len(xticklabels):
            ax.set_xticklabels(xticklabels, rotation=xticklabel_ro)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if 'left' in spines or 'right' in spines:
        if yticks == 'none':
            ax.set_yticks([])
        elif not yticks == 'auto':
            ax.set_yticks(yticks)
        if yticks_minor == 'none':
            ax.set_yticks([], minor=True)
        elif not yticks_minor == 'auto':
            ax.set_yticks(yticks_minor, minor=True)
        if len(yticklabels):
            ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
    else:
        # no yaxis ticks
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
    elif 'top' in spines:
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
    else:
        # no xaxis ticks
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            # ax.set_xticklabels([])

def plot_x_scale_text(ax, bar_length, text, x_text_lefter=None, y_text_lower=None, color='black', lw=2, **kwargs):
    ylim = ax.get_ylim()
    y_length = ylim[-1] - ylim[0]
    if not x_text_lefter:
        x_text_lefter = 0.1
    if not y_text_lower:
        y_text_lower = 0.1
    xlim = ax.get_xlim()
    x_length = xlim[-1] - xlim[0]
    x_text = xlim[-1] - bar_length/2.0 - x_length * x_text_lefter
    y_text = ylim[0] + y_length * y_text_lower
    y_bar = y_text - y_length * 0.01
    x0 = x_text - bar_length / 2.0
    x1 = x_text + bar_length / 2.0
    ax.plot([x0, x1], [y_bar, y_bar], c=color, lw=lw, solid_capstyle='butt', clip_on=False)
    ax.text(x_text, y_text, text, color=color, ha='center', va='bottom', clip_on=False, **kwargs)

def plot_y_scale_text(ax, bar_length, text, x_text_right=None, y_text_upper=None, color='black',  **kwargs):
    ylim = ax.get_ylim()
    y_length = ylim[-1] - ylim[0]
    xlim = ax.get_xlim()
    x_length = xlim[-1] - xlim[0]
    if not x_text_right:
          x_text_right = 0.1
    if not y_text_upper:
        y_text_upper = 0.05
    x_text = xlim[0] + x_length * x_text_right
    y_text = ylim[0] + bar_length / 2.0 + y_length * y_text_upper
    x_bar = x_text + x_length * 0.01
    y0 = y_text - bar_length / 2.0
    y1 = y_text + bar_length / 2.0
    ax.plot([x_bar, x_bar], [y0, y1], c=color, lw=1, solid_capstyle='butt', clip_on=False)
    ax.text(x_text, y_text, text, color=color, clip_on=False, **kwargs)

def plot_colormap(ax, colormap=plt.cm.jet, reverse_cm=True, ylim=[0,256], yticks=[0,256], yticklabels=[], ylabel='ylabel', label_rotation=0, label_pos='right', label_axeratio=False):
    # plot colormap on a self-defined axis

    gradient = np.linspace(0, 1 - math.pow(10,-4), 256)
    gradient = np.vstack((gradient, gradient)).T[::-1,:] if reverse_cm else np.vstack((gradient, gradient)).T
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.tick_right()
    ax.set_ylabel(ylabel, rotation=label_rotation, va='center', ha='left')
    ax.yaxis.set_label_position(label_pos)
    if label_axeratio:
        ax.get_yaxis().set_label_coords(label_axeratio, .5)
    ax.set_xticks([])
    ax.set_ylim(ylim)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params('both', length=0, width=0, which='major')

def large_ax(size=(20,20)):
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    return ax

def save(suffix, rec=None, parentdir=True, exts=['png', 'pdf'], id=False, dpi=300,
         facecolor='none'):
    if parentdir:
        folder = './'
    else:
        folder = rec.folder
    directories = ['%s%s/' %(folder, ext) for ext in exts]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    for ext in exts:
        if id and rec:
            flyid = rec.name
        else:
            flyid = ''
        plt.savefig('%s%s/%s%s.%s' %(folder, ext, flyid, suffix, ext), bbox_inches='tight',
                    transparent=True, dpi=dpi, facecolor=facecolor)

def special_color(special_color_nums, special_colors, i_col):
    if i_col in special_color_nums:
        if len(special_color_nums) == len(special_colors):
            color = special_colors[special_color_nums.index(i_col)]
        else:
            color = special_colors[0]
    else:
        color = False
    return color

def legend_group_number(group_info):
    # find out which group has the largest number of subgroups, use it to plot legend
    i_max = 0
    max_num = 0
    for igroup, sublist in enumerate(group_info):
        if len(sublist) > max_num:
            i_max = igroup
            max_num = len(sublist)
    return i_max

def errorbar_lc_xss(xss=[], ax=False, group_info=[[]],
                    col_width=0.4, subgroup_gap=0.1, group_gap=1, plot_scale=1, margin_to_yaxis=.5, figsize_height=5,
                    group_labels=[], colors=[grey2], special_color_nums=[], special_colors=[],
                    yerr_type='sem', ylabel='', yticks=np.array([0, 1]), ylim=[0,1], yticks_minor=[], fs=15, ms_indi=5,
                    alpha_indi=1,
                    noyaxis=False, ro_x=45, label_axeratio=True, rotate_ylabel=False, show_legend=False, legend_list=[],
                    legend_ncol=2, leg_loc='upper right', bbox_to_anchor=False, fs_legend=12, connecting_mean_list=[],
                    connecting_indi_list=[], ttest_list=[], test_type='student', plot_bar=True, alpha_bar=.2,
                    subgroup_samefly=False, show_flynum=True, legend_frameon=False, yaxis_ratio=1,
                    show_nancol=True, show_xlabel=True, hlines=[]):
    """
    second version of error bar plot, after function errorbar_lc_xsss.
    each plot is consisted of multiple groups, which is consisted of multiple subgroups (columns)
    group_labels only labels each group
    legend and color applies within each group, i.e, legends and colors are the same across groups, but different across subgroups
    :param xss:
        # 'x' represents each sample dot
        # 'xs' represents each subgroup, i.e, single columns
        # 'xss' represents all columns
    :param group_info: list of list, each sublist consists the seq # of each column to be in the same group
    :param colors: list, enumerate within each group columns
    :param special_color_nums: number of single column sequences
    :param special_colors: if len(special_colors) == len(special_color_nums), 1 to 1, otherwise, all use same spe color
    :param connecting_mean_list: list of list, each sublist consists the seq # of each column to be connected, one to next
    :param connecting_indi_list: same as connecting_means
    :return: ax
    """

    # calculate plotting parameters
    num_group = len(group_info)
    num_col = len([item for sublist in group_info for item in sublist])
    num_legend = len(legend_list)
    if num_col != len(xss):
        raise NameError("Error: length of xss does not match group_info, in function: ph.errorbar_lc_xss")
    xlim = [0, margin_to_yaxis + (num_col+2)*col_width + (num_group-1)*group_gap + (num_col-num_group)*subgroup_gap]
    y_xlabel = ylim[0] - (ylim[1] - ylim[0]) * .15
    y_xlabel_line = ylim[0] - (ylim[1] - ylim[0]) * .14
    y_n = ylim[0] - (ylim[1] - ylim[0]) * .1
    i_legend_group = legend_group_number(group_info)    # find out which group has the largest number of subgroups, use it to plot legend


    # setup plotting structure
    set_fontsize(fs)
    if not ax:
        fig = plt.figure(1, (xlim[-1]*plot_scale, figsize_height))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
    xmeans = []  # for saving the xy coordinates of the means
    ymeans = []  # for connecting_means
    indi_xpositions = [] # for saving the x coordinates of the individuals


    # plot individual fly and mean data
    for i_group in range(num_group):
        col_nums = group_info[i_group]

        for i_subgroup, i_col in enumerate(col_nums):

            # calculate plotting position, colors, etc.
            x_center = margin_to_yaxis + i_col*col_width + i_group*group_gap + (i_col-i_group)*subgroup_gap + .5*col_width
            _color = special_color(special_color_nums, special_colors, i_col)
            color = _color if _color else colors[i_subgroup % len(colors)]
            indi_xpositions.append([])

            # calculate plotting stats
            n_fly = len(xss[i_col])
            y_mean = np.nanmean(xss[i_col])
            if yerr_type == 'std':
                yerr = np.nanstd(xss[i_col])
            elif yerr_type == 'sem':
                yerr = np.nanstd(xss[i_col]) / np.sqrt(n_fly)
            elif yerr_type == 'CI95':
                yerr = fc.confidence_interval(xss[i_col], confidence=0.95)
            else:
                raise NameError("Not recognized y_type!")
            xmeans.append(x_center)
            ymeans.append(y_mean)

            # plot individual fly marker
            for i in range(n_fly):
                x_random = x_center + (random.random() - .5) * col_width * 1
                indi_xpositions[-1].append(x_random)
                if plot_bar:
                    ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec='black', mew=1, mfc='none', alpha=alpha_indi)
                else:
                    if i == 0 and i_group == 0 and len(legend_list):
                        ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec=color, mew=1, mfc='none',
                                label=legend_list[i_subgroup], alpha=alpha_indi, zorder=1)
                    else:
                        ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec=color, mew=1, mfc='none', alpha=alpha_indi, zorder=1)

            # plot mean
            ax.errorbar(x_center, y_mean, yerr=yerr, color='black', fmt='_', ms=5, capsize=5, mew=1.5,
                            ecolor='black', elinewidth=1, zorder=2)
            # ax.errorbar(x_center, y_mean, yerr=yerr, color='black', fmt='_', ms=7, capsize=5, mew=1,
            #             ecolor='black', elinewidth=2, zorder=2)

            # plot bar
            if plot_bar:
                if show_legend and i_group == i_legend_group and i_subgroup < num_legend and len(legend_list):
                    ax.bar(x_center, y_mean, width=col_width, align='center', facecolor=color, edgecolor='none', linewidth=0,
                           label=legend_list[i_subgroup], alpha=alpha_bar, zorder=1)
                else:
                    ax.bar(x_center, y_mean, width=col_width, align='center', facecolor=color, edgecolor='none', linewidth=0,
                           alpha=alpha_bar, zorder=1)

            # plot number of flies
            if show_flynum:
                if subgroup_samefly:
                    if i_subgroup == 0:
                        x_center_nfly_start = margin_to_yaxis + group_info[i_group][0] * col_width + \
                                              i_group * group_gap + (group_info[i_group][0] - i_group) * subgroup_gap
                        x_center_nfly_length = len(group_info[i_group]) * col_width + \
                                               (len(group_info[i_group]) - 1) * subgroup_gap
                        x_center_nfly = x_center_nfly_start + x_center_nfly_length / 2.
                        ax.text(x_center_nfly, y_n, '%s' % len(xss[i_col]), ha='center', va='bottom')
                else:
                    ax.text(x_center, y_n, '%s' % len(xss[i_col]), ha='center', va='bottom')
                if i_col == 0:
                    ax.text(-subgroup_gap, y_n, 'n', ha='center', va='bottom')

        # plot xlabels
        if show_xlabel and (not np.isnan(y_mean) or show_nancol):
            x_xlabel_start = margin_to_yaxis + group_info[i_group][0] * col_width + i_group * group_gap \
                                 + (group_info[i_group][0] - i_group) * subgroup_gap
            x_xlabel_length = len(group_info[i_group]) * col_width + (len(group_info[i_group]) - 1) * subgroup_gap
            if ro_x == 0:
                x_xlabel_center = x_xlabel_start + x_xlabel_length / 2.
                ax.text(x_xlabel_center, y_xlabel, group_labels[i_group], ha='center', va='top', rotation=ro_x)
            else:
                x_xlabel_right = x_xlabel_start + x_xlabel_length
                ax.text(x_xlabel_right, y_xlabel, group_labels[i_group], ha='right', va='top', rotation=ro_x)
            x_xlabelline_left = x_xlabel_start + col_width/4.
            x_xlabelline_right = x_xlabel_start + x_xlabel_length - col_width/4.
            ax.plot([x_xlabelline_left, x_xlabelline_right], [y_xlabel_line, y_xlabel_line], color='black', lw=2, clip_on=False)

    # trim plot axes
    if noyaxis:
        adjust_spines(ax, [], lw=.4, xticks='none', yticks=[], yticks_minor=[], xlim=xlim, ylim=ylim)
        ylabel = ''
    elif yaxis_ratio == 1:
        adjust_spines(ax, ['left'], lw=.4, xticks='none', yticks=yticks, yticks_minor=yticks_minor, xlim=xlim, ylim=ylim)
    else:
        adjust_spines(ax, ['left'], lw=.4, xticks='none', yticks=yticks, yticks_minor=yticks_minor, xlim=xlim, ylim=ylim,
                      yticklabels=(np.array(yticks) * yaxis_ratio).astype('str'))

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)

    if ylabel:
        if label_axeratio:
            if rotate_ylabel:
                ax.set_ylabel(ylabel, rotation='horizontal', ha='center')
            else:
                ax.set_ylabel(ylabel)
            if type(label_axeratio) is float:
                ax.get_yaxis().set_label_coords(label_axeratio, .5)
            else:
                ax.get_yaxis().set_label_coords(-.1, 0.5)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            tx = xlim[0] - 1.9
            ty = ylim[0] + (ylim[-1] - ylim[0]) * .5
            ax.text(tx, ty, ylabel, ha='center', va='center', fontsize=fs, rotation=90)

    if show_legend:
        if bbox_to_anchor:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, bbox_to_anchor=bbox_to_anchor, ncol=legend_ncol, frameon=legend_frameon)
        else:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, ncol=legend_ncol, frameon=legend_frameon)

    # additional plot features
    if len(connecting_mean_list):
        for cols in connecting_mean_list:
            for _i in range(len(cols)-1):
                col0 = cols[_i]
                col1 = cols[_i + 1]
                ax.plot([xmeans[col0], xmeans[col1]], [ymeans[col0], ymeans[col1]], color=grey5, lw=2, alpha=1, zorder=1)

    if len(connecting_indi_list):
        for cols in connecting_indi_list:
            # check all cols have same number of flies
            if len(np.unique([len(xss[_col]) for _col in cols])) == 1:
                for _i in range(len(cols) - 1):
                    for _j in range(len(xss[cols[_i]])):
                        col0 = cols[_i]
                        col1 = cols[_i + 1]
                        ax.plot([indi_xpositions[col0][_j], indi_xpositions[col1][_j]], [xss[col0][_j], xss[col1][_j]], color=grey2, lw=1, alpha=.5, zorder=1)

    if len(ttest_list):
        for cols in ttest_list:
            for _i in range(len(cols) - 1):
                col0 = cols[_i]
                col1 = cols[_i + 1]
                x0, x1 = xmeans[col0], xmeans[col1]
                y_bar = ylim[0] + (ylim[-1] - ylim[0]) * 0.83
                x_marker = (x0 + x1) / 2.
                y_marker = ylim[0] + (ylim[-1] - ylim[0]) * 0.84

                ttest_a = np.array(xss[col0])[~np.isnan(xss[col0])]
                ttest_b = np.array(xss[col1])[~np.isnan(xss[col1])]
                if test_type == 'student':
                    _, p2 = stats.ttest_ind(ttest_a, ttest_b)
                    p_value = p2
                elif test_type == 'mannwhitneyu':
                    _, p_value = stats.mannwhitneyu(ttest_a, ttest_b)

                if p_value >= 0.05:
                    p_value_marker = 'ns'
                elif p_value >= 0.01:
                    p_value_marker = '*'
                elif p_value >= 0.001:
                    p_value_marker = '**'
                elif p_value >= 0:
                    p_value_marker = '***'
                else:
                    p_value_marker = 'err'
                # print p_value
                ax.plot([x0+col_width/4., x1-col_width/4.], [y_bar, y_bar], lw=2, color='black')
                ax.text(x_marker, y_marker, p_value_marker, ha='center', va='bottom', fontsize=fs, color='black')
                print(p_value)

    if len(hlines):
        for hline in hlines:
            ax.axhline(0, ls='--', lw=1, c=grey9, alpha=.5, zorder=1)