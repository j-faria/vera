from functools import partialmethod
import numpy as np
from matplotlib import pyplot as plt
from astropy.time import Time


def _dates_on_top(ax):
    ax1 = ax.twiny()
    times = Time(ax.get_xticks() + 24e5, format='jd')
    dates = [
        '/'.join(d.split('T')[0].split('-')[::-1]) for d in times.isot
    ]
    ax1.set_xticklabels(dates, rotation=45, fontsize=7, ha='center')
    return ax1

from .utils import styleit

@styleit
def plot(self,
         include_sigma_clip=False,
         ax=None,
         show_dates=False,
         show_today=False,
         show_fibers=False,
         instruments=None,
         color=None,
         legend=True,
         right_ticks=False,
         tooltips=True,
         offset=0,
         remove_50000=False,
         **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure

    instrument_labels = self._instrument_labels
    m = self.mask

    toffset = 50000 if remove_50000 else 0
    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    all_lines = []
    for i in np.unique(self.obs):
        if instruments:  # only plot some instruments
            if self.instruments[int(i) - 1] not in instruments:
                continue
        ind = self.obs[m] == i

        lines, caplines, barlines = \
            ax.errorbar(self.time[m][ind] - toffset, self.vrad[m][ind] + offset,
                        self.svrad[m][ind], color=color, **kwargs,
                        label=instrument_labels[int(i - 1)])
        all_lines.append(lines)

        if include_sigma_clip:
            ind = self.obs[~m] == i
            ax.errorbar(self.time[~m][ind] - toffset,
                        self.vrad[~m][ind] + offset,
                        self.svrad[~m][ind],
                        fmt='x',
                        color='r',
                        alpha=0.4,
                        capsize=0,
                        label=self.instruments[int(i) - 1])

    if tooltips:
        from mpldatacursor import datacursor

        def tooltip_formatter(**kwargs):
            x, y = kwargs['x'], kwargs['y']

            # this doesn't work when observations from different instruments
            # (or pipelines) have the same time
            # ind = np.where(self.time == x)[0][0]
            # this seems to work
            find_x = np.where(self.time == x)
            find_y = np.where(self.vrad == y)
            ind = np.intersect1d(find_x, find_y)[0]

            e = self.svrad[ind]
            # i = self.instruments[int(self.obs[ind]) - 1]
            dt = self.datetimes[ind]
            tt = f'⊢ {dt} ⊣\n'
            tt += f'index: {ind}\n'
            tt += 'BJD: ' + f'{x:.5f}' + '\n'
            tt += f'RV[{self.units}]: ' + f'{y:.4f} ± {e:.4f}'
            return tt

        c = datacursor(
            all_lines,
            bbox=dict(fc='white', alpha=0.8),  #usetex=True,
            formatter=tooltip_formatter,
            ha='left',
            fontsize=8)

        # allow pressing "r" to remove selected point
        def on_keypress(event):
            if c._last_event is None:
                return

            if event.key == 'r':
                print(f'Pressed "{event.key}".')
                props = c.event_info(c._last_event)
                # same problem as above
                # ind = np.where(self.time == props['x'])[0][0]
                # seems to work
                find_x = np.where(self.time == props['x'])
                find_y = np.where(self.vrad == props['y'])
                ind = np.intersect1d(find_x, find_y)[0]
                # print(f'Removing point with index {ind}')
                self.remove_point(ind)
                plt.close(fig)
                self.plot()

        fig.canvas.mpl_connect('key_press_event', on_keypress)

    if show_fibers:
        ax.axvline(self.technical_intervention, color='k', ls='--', lw=2)
        ax.grid(axis='y')

        before = self.time < self.technical_intervention
        after = self.time > self.technical_intervention
        justESP = self.instruments.size == 1 and 'ESPRESSO' in self.instruments
        if before.any() and after.any() and justESP:
            ax.hlines(y=self.vrad[before & self.mask].mean(),
                      xmin=self.technical_intervention - 5,
                      xmax=self.technical_intervention + 1,
                      color='r')
            ax.hlines(y=self.vrad[after & self.mask].mean(),
                      xmin=self.technical_intervention - 1,
                      xmax=self.technical_intervention + 5,
                      color='r')

    if remove_50000:
        ax.set(xlabel='BJD - 2450000 [days]', ylabel=f'RV [{self.units}]')
    else:
        ax.set(xlabel='BJD - 2400000 [days]', ylabel=f'RV [{self.units}]')

    if legend:
        ax.legend()
    if right_ticks:
        ax.tick_params(right=True, labelright=True)

    # add today line, now that the limits are all done
    if show_today:
        y1, y2 = ax.get_ylim()
        h = 0.1 * abs(y2 - y1)
        today = Time.now().jd - 24e5
        ax.vlines([today], ymin=y1, ymax=y1 + h, color='g', alpha=0.6)
        ax.set_ylim(y1, y2)

    if show_dates:
        _ = _dates_on_top(ax)

    return ax, None


@styleit
def plot_both(self,
              contrast=False,
              rhk=False,
              sindex=False,
              naindex=False,
              haindex=False,
              ccf_asym=False,
              show_fibers=False,
              **kwargs):
    """ Plot RVs together with other indicators (the FWHM, by default). """

    fig, (ax, ax1) = plt.subplots(ncols=1,
                                  nrows=2,
                                  constrained_layout=True,
                                  sharex=True)

    self.plot(ax=ax, **kwargs)

    # if shown on the first axis, don't show on the second
    kwargs.pop('show_dates', None)

    # no legend on the second axis
    kwargs.setdefault('legend', False)

    if rhk:
        self.plot_rhk(ax=ax1, **kwargs)
    elif contrast:
        self.plot_contrast(ax=ax1, **kwargs)
    else:
        self.plot_fwhm(ax=ax1, **kwargs)

    ax.set(xlabel='Time [days]', ylabel=f'RV [{self.units}]')
    ax1.set(xlabel='Time [days]')
    if contrast:
        ax1.set_ylabel('Contrast [%]')
    elif rhk:
        ax1.set_ylabel("R'hk")
    elif sindex:
        ax1.set_ylabel("S index")
    else:
        ax1.set_ylabel(f'FWHM [{self.units}]')

    if show_fibers:
        ax.axvline(self.technical_intervention, color='k', ls='--', lw=2)
        ax1.axvline(self.technical_intervention, color='k', ls='--', lw=2)

    # ax.legend()
    return fig, (ax, ax1)


@styleit
def plot_indicator(self,
                   value,
                   error,
                   label,
                   include_sigma_clip=False,
                   ax=None,
                   show_dates=False,
                   show_today=False,
                   show_fibers=False,
                   instruments=None,
                   latest_pipeline=True,
                   color=None,
                   legend=True,
                   right_ticks=False,
                   tooltips=True,
                   **kwargs):

    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    instrument_labels = self._instrument_labels
    m = self.mask

    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('capsize', 0)
    kwargs.setdefault('ms', 4)

    try:
        value = getattr(self, value)
    except AttributeError:
        exec('out = ' + value, globals(), locals())
        value = locals()['out']

    if error is None:
        error = np.zeros_like(value)
    else:
        error = getattr(self, error)

    all_lines = []
    for i in np.unique(self.obs):
        if instruments:  # only plot some instruments
            if self.instruments[int(i) - 1] not in instruments:
                continue
        ind = self.obs[m] == i

        lines, caplines, barlines = \
            ax.errorbar(self.time[m][ind], value[m][ind],
                        error[m][ind], color=color, **kwargs,
                        label=instrument_labels[int(i - 1)])
        all_lines.append(lines)

        if include_sigma_clip:
            ind = self.obs[~m] == i
            ax.errorbar(self.time[~m][ind],
                        value[~m][ind],
                        error[~m][ind],
                        fmt='x',
                        color='r',
                        alpha=0.4,
                        capsize=0,
                        label=self.instruments[int(i) - 1])

    if tooltips:
        from mpldatacursor import datacursor

        def tooltip_formatter(**kwargs):
            x, y = kwargs['x'], kwargs['y']
            ind = np.where(self.time == x)[0][0]
            e = error[ind]
            # i = self.instruments[int(self.obs[ind]) - 1]
            dt = self.datetimes[ind]
            tt = f'⊢ {dt} ⊣\n'
            tt += f'index: {ind}\n'
            tt += 'BJD: ' + f'{x:.5f}' + '\n'
            tt += f'RV[{self.units}]: ' + f'{y:.4f} ± {e:.4f}'
            return tt

        datacursor(
            all_lines,
            bbox=dict(fc='white', alpha=0.8),  #usetex=True,
            formatter=tooltip_formatter,
            ha='left',
            fontsize=8)

    # add today line, now that the limits are all done
    if show_today:
        y1, y2 = ax.get_ylim()
        h = 0.1 * abs(y2 - y1)
        today = Time.now().jd - 24e5
        ax.vlines([today], ymin=y1, ymax=y1 + h, color='g', alpha=0.6)
        ax.set_ylim(y1, y2)

    if show_dates:
        ax1 = ax.twiny()
        ax1.plot(self.time[m], self.vrad[m], alpha=0)

        # times = Time(self.time[m] + 24e5, format='jd')
        times = Time(ax.get_xticks() + 24e5, format='jd')

        dates = [
            '/'.join(d.split('T')[0].split('-')[::-1]) for d in times.isot
        ]
        # ax1.set_xticks(self.time[m])
        # ax1.set_xticks(ax.get_xticks())
        ax1.set_xticklabels(dates, rotation=45, fontsize=7, ha='center')

    ax.set(xlabel='Time [days]', ylabel=label)
    if legend:
        ax.legend()
    if right_ticks:
        ax.tick_params(right=True, labelright=True)

    return ax


plot_fwhm = partialmethod(plot_indicator, 'fwhm', 'efwhm', 'FWHM [m/s]')

plot_contrast = partialmethod(plot_indicator, 'contrast', 'econtrast',
                              'CCF contrast [%]')

plot_sindex = partialmethod(plot_indicator, 'sindex', 'esindex', 'S index')

plot_rhk = partialmethod(plot_indicator, 'rhk', 'erhk', r"log R'$_{\rm HK}$")

plot_berv = partialmethod(plot_indicator, 'berv', None, 'BERV')
