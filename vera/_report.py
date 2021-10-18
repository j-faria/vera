import inspect
from copy import copy
from itertools import cycle

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time

from .utils import get_artist_location, get_function_call_str
from .utils import blue
from .query_periods import ESO_periods


def report(self,
           save=None,
           dl=False,
           plot_fwhm=True,
           plot_rhk=False,
           show_today=True,
           show_sigmaclipped=False,
           show_other_names=True,
           show_ESO_periods=False,
           secacc=True,
           show_dates=False,
           show_visibility=True,
           show_comments=True,
           error_ratio=True,
           score=False,
           show_fibers=False,
           instruments=None):

    # first, check if we need to run any data treatment
    if not self._ran_treatment:
        self._run_treament_steps()

    # try to remove the secular acceleration
    self.secular_acceleration(plot=False)

    # size = 11.69, 8.27 # for landscape
    size = 8.27, 11.69
    fig = plt.figure(figsize=size, constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.1)
    # fig.suptitle(self.star)
    gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[2, 2, 1, 2, 0.1])

    # first row, all columns
    ax1 = plt.subplot(gs[0, :])

    nb = ' (binned)' if self.is_binned else ''
    sa = ' (SA)' if self._removed_secular_acceleration else ''

    title = f'{self.star}{nb}{sa}'
    ax1.set_title(title, loc='left', fontsize=12)
    # ax1.set_title(r"\href{http://www.google.com}{link}", color='blue',
    #               loc='center')

    title = f'{self.spectral_type}'

    if show_other_names:
        othernames = [n.replace(' ', '') for n in sorted(self.othernames)]
        # title = f'pr:{self.priority}  {self.spectral_type}    '
        title += '  ' + ' | '.join(othernames)

    ax1.set_title(title, loc='right', fontsize=10)

    ax1, axdates = self.plot(ax=ax1,
                             include_sigma_clip=show_sigmaclipped,
                             show_dates=show_dates,
                             show_fibers=show_fibers,
                             instruments=instruments)
    ax1.get_legend().remove()

    # second row, all columns excluding last
    ax2 = plt.subplot(gs[1, :-1])
    # third row, all columns excluding last
    ax3 = plt.subplot(gs[2, :-1])
    if plot_fwhm or plot_rhk:
        self.gls(ax=ax2, frequency=False, bootstrap=False, HZ=True)
        # ax2.legend()
        if plot_fwhm:
            self.plot_fwhm(ax=ax3,
                           include_sigma_clip=show_sigmaclipped,
                           instruments=instruments,
                           show_fibers=show_fibers)
        elif plot_rhk:
            self.plot_fwhm(ax=ax3,
                           include_sigma_clip=show_sigmaclipped,
                           instruments=instruments,
                           show_fibers=show_fibers)
        ax3.get_legend().remove()
    else:
        self.gls(axs=(ax2, ax3))

    #
    ax4 = plt.subplot(gs[-2, 0])
    ax4c = plt.subplot(gs[-1, 0])
    cb = self.correlate(ax=ax4, axcb=ax4c, instruments=instruments)
    cb.set_ticks([])

    #
    ax5 = plt.subplot(gs[-2, 1:])
    if dl and self.stellar_mass != 0:
        self.detection_limits(ax=ax5, current=True)
    else:
        if show_visibility:
            self.visibility(ax=ax5)
        else:
            ax5.set_title('detection limits')
            ax5.set(xlabel='Period [d]', ylabel=r'Mass [M$_\oplus$]')

    if not show_visibility:
        ax5.axvspan(*self.HZ, color='g', alpha=0.2)
        if self.known_planets.P is not None:
            y1, y2 = ax5.get_ylim()
            h = 0.1 * abs(y2 - y1)
            ax5.vlines(self.known_planets.P,
                       ymin=y1,
                       ymax=y1 + h,
                       color='m',
                       alpha=0.6)

    # if self.instruments.size > 1 and error_ratio:
    # ax6 = plt.subplot(gs[2, 2])
    # # ax6.axis('off')
    # ni = self.instruments.size

    # ax6.plot(
    #     range(1, ni + 1), [i.error for i in self.each], 'k-',
    #     alpha=0.2)
    # ax6.plot(
    #     range(1, ni + 1), [i.rms for i in self.each], 'k-', alpha=0.2)
    # for i, inst in enumerate(self.each):
    #     l = ax6.plot(i + 1, inst.error, 'o')
    #     l = ax6.plot(i + 1, inst.rms, '^', color=l[0].get_color())

    # # try:
    # #     ax6.plot(1, self.each.ESPRESSO.error, 'C0o')
    # #     ax6.plot(1, self.each.ESPRESSO.rms, 'C0^')
    # # except AttributeError:
    # #     pass

    # ax6.set_xticks(range(1, ni + 1))
    # ax6.set_xticklabels(self._instrument_labels, rotation=20, fontsize=6)
    # # ax6.set_title(r'$\bar{\sigma}_{RV}$ and $RV_{\rm RMS}$',
    # #               loc='right', fontsize=10)
    # ax6.set(ylim=(-0.3, np.ceil(ax6.get_ylim()[1])), ylabel='[m/s]')
    # ax6.grid(True, which='major', axis='y')
    # handles = [
    #     Line2D([0], [0], marker='^', color='k', lw=0,
    #            label=r'$\bar{\sigma}_{RV}$'),
    #     Line2D([0], [0], marker='o', color='k', lw=0,
    #            label=r'$RV_{\rm RMS}$'),
    # ]
    # ax6.legend(handles=handles, fontsize=10, handletextpad=0,
    #            borderpad=0.2)
    # # for i, inst in enumerate(self.each):
    # #     ax6.plot(i+1, inst.error, 'mo')
    # #     ax6.plot(i+1, inst.rms, 'go')

    # if score:
    #     ncols = 6
    #     # data = np.fromiter(self.score.values(), dtype=np.float)
    #     # data = data.reshape(1, -1).round(2)
    #     data = [list(self.score.values())]
    #     data = [[round(d, 2)
    #              for d in data[0]]]  # skips the ints, apparently
    #     # print(data)
    #     collabel = ("$S_{ST}$", "$S_{\sigma}$", "$S_{HZ}$",
    #                 "$S^{HZ}_{Prot}$", "$S_{pl}$", "total")
    #     # bbox = [1.1, 0.4, 0.5, 0.5] # right of ax3
    #     bbox = [-0.2, 1.08, 1.25, 0.35]
    #     table = ax6.table(cellText=data, colLabels=collabel, loc='top',
    #                       colWidths=ncols * [0.2], cellLoc='center',
    #                       bbox=bbox, colColours=(ncols - 1) * ['w'] +
    #                       ['tomato'])  #, edges='open')
    #     # ax6 = plt.subplot(gs[2, 2])

    #
    axt = plt.subplot(gs[1, -1])
    axt.text(0, 1, 'instruments:')
    hand, lab = ax1.get_legend_handles_labels()
    leg = axt.legend(hand,
                     lab,
                     loc='upper center',
                     ncol=2,
                     borderaxespad=0.,
                     borderpad=0.3,
                     bbox_to_anchor=(0.5, 0.98),
                     handletextpad=0,
                     columnspacing=0.4)

    *_, legend_bottom, _ = get_artist_location(leg, axt, fig)

    t = axt.text(0, 0.63, 'number of observations:', va='bottom')
    *_, text_bottom, _ = get_artist_location(t, axt, fig)

    # longest_name
    ln = max(len(inst) for inst in self.instruments)
    items = self.N_complete.items()
    msg = '\n'.join([f'{k:{ln}s} : {n:5d} ({nr})' for k, (n, nr) in items])

    t = axt.text(0.2, text_bottom - 0.01, msg, family='monospace', va='top')
    *_, text_bottom, _ = get_artist_location(t, axt, fig)

    t = axt.text(0,
                 text_bottom - 0.01,
                 va='top',
                 s=f'weighted rms: \n    {self.rms:.3f} {self.units}')
    *_, text_bottom, _ = get_artist_location(t, axt, fig)

    t = axt.text(0,
                 text_bottom,
                 va='top',
                 s=f'average error bar: \n    {self.error:.3f} {self.units}')
    *_, text_bottom, _ = get_artist_location(t, axt, fig)

    # t = axt.text(0, text_bottom - 0.02, self.comments, va='top', wrap=True)

    axt.axis('off')

    if any(['CORALIE' in i for i in self.instruments]):
        print('coralie ccf coming')
        # self.enable_chromatic()
        ax6 = plt.subplot(gs[2, 2])
        self.cRV.plot_ccfs(ax=ax6)
        # ax6.plot(self.cRV.I[0].rv, self.cRV.ccf[0])
        ax6.set_yticklabels([])

    #
    # fig.text(0.99,
    #          0.005,
    #          f'CC-BY © João Faria {Time.now().value.year}',
    #          fontsize=10,
    #          color='gray',
    #          ha='right',
    #          va='bottom',
    #          alpha=0.5)

    # y1, _ = ax1.get_ylim()
    # x1, _ = ax1.get_xlim()
    try:
        msg = f'DACE queried on {self._lastDACEquery} UTC'
        ax1.text(0.01,
                 0.02,
                 msg,
                 fontsize=8,
                 rotation=0,
                 color='gray',
                 ha='left',
                 va='bottom',
                 alpha=0.5,
                 transform=ax1.transAxes)
    except AttributeError:
        pass

    ax1.text(0.95,
             0.02,
             self.star,
             fontsize=8,
             rotation=0,
             color='gray',
             ha='right',
             va='bottom',
             alpha=0.1,
             transform=ax1.transAxes)

    if instruments:
        msg = f'showing data from {",".join(instruments)} only'
        ax1.text(0.01,
                 0.08,
                 msg,
                 fontsize=8,
                 rotation=0,
                 color='gray',
                 ha='left',
                 va='bottom',
                 alpha=0.5,
                 transform=ax1.transAxes)

        msg = 'periodogram of full dataset'
        ax2.text(0.01,
                 0.92,
                 msg,
                 fontsize=8,
                 rotation=0,
                 color='gray',
                 ha='left',
                 va='bottom',
                 alpha=0.5,
                 transform=ax2.transAxes)

    # add today line to ax1, now that the limits are all done
    if show_today:
        y1, y2 = ax1.get_ylim()
        h = 0.1 * abs(y2 - y1)
        today = Time.now().jd - 24e5
        ax1.vlines([today], ymin=y1, ymax=y1 + h, color='g', alpha=0.6)
        ax1.set_ylim(y1, y2)

    # add ESO periods to ax1
    if show_ESO_periods:
        y1, y2 = ax1.get_ylim()
        h = 0.1 * abs(y2 - y1)
        iteralpha = cycle([0.1, 0.4])
        for i, v in enumerate(ESO_periods.values()):
            # if i < 2: continue
            if i > 3:
                break
            ax1.axvspan(*v[2], ymin=0.8, alpha=iteralpha.__next__(), color='b')
        # today = Time.now().jd - 24e5
        # ax1.vlines([today], ymin=y1, ymax=y1 + h, color='g', alpha=0.6)
        ax1.set_ylim(y1, y2)

    if show_dates:
        axdates.set_xlim(ax1.get_xlim())

    if save is not None:
        if save is True:
            save = f'report_{"".join(self.star.split())}.pdf'

        with PdfPages(save) as pdf:
            # we don't want to add this call to _reproduce, only to note
            reproduce = copy(self._reproduce)
            reproduce.append(
                get_function_call_str(inspect.currentframe(), top=False))
            reproduce = '\n'.join(reproduce)
            note = f'To reproduce:\n{reproduce}'
            pdf.attach_note(note, positionRect=[5, 15, 20, 30])

            # comments on this star?
            if self.comments and show_comments:
                pos = [560, 450, 575, 465]
                pdf.attach_note(self.comments.replace('\\', '\n'),
                                positionRect=pos)

            if self.verbose:
                print(blue | 'Saving report to', save)
            pdf.savefig(fig)

        plt.close('all')
        # os.system(f'evince {save} &')
