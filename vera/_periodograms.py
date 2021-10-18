from functools import partial, partialmethod

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    from gatspy import periodic
except ImportError:
    raise ImportError('Please, pip install gatspy')

from astropy.timeseries import LombScargle
from .utils import info, red
from .stat_tools import wrms, false_alarm_level_gatspy


def frequency_grid(self,
                   plow=None,
                   phigh=None,
                   samples_per_peak=5,
                   nmax=50000):

    if plow is None:
        plow = 0.5
    if phigh is None:
        phigh = 2 * self.time.ptp()

    #! HACK
    return np.linspace(1 / phigh, 1 / plow, 20000)

    day2sec = 86400
    sec2day = 1 / day2sec

    ## chosing the maximum frequency
    self.extract_from_DACE_data('texp')
    texp = self.texp[self.mask].mean()
    windowing_limit = 0.5 / (texp * sec2day)
    # print('windowing_limit:', windowing_limit, 'day^-1')

    # observation times are recorded to D decimal places
    time_precision = len(str(self.time[0]).split('.')[1])
    time_precision_limit = 0.5 * 10**time_precision
    # print('precision_limit:', time_precision_limit, 'day^-1')

    fmax = min(windowing_limit, time_precision_limit)

    ## choosing the minimum frequency (easy)
    timespan = self.time[self.mask].ptp()
    fmin = 1 / timespan

    Neval = int(samples_per_peak * texp * fmax)
    # print('Neval:', Neval)
    Neval = min(Neval, nmax)

    freq = np.linspace(fmin, fmax, Neval)
    return freq


def window_function(self, plot=True, frequency=True, norm=None, **kwargs):
    """ Calculate the window function of the sampling times. """
    if self.time[self.mask].size < 3:
        print(red | 'Cannot calculate periodogram! Too few points?')
        return

    if norm is None:
        norm = 'standard'
    m = self.mask
    t, e = self.time[m], self.svrad[m]
    ls = LombScargle(t, np.ones_like(t), e, fit_mean=False, center_data=False,
                     normalization=norm)

    minf = kwargs.pop('minf', None)
    minf = kwargs.pop('minimum_frequency', minf)
    maxf = kwargs.pop('maxf', None)
    maxf = kwargs.pop('maximum_frequency', maxf)

    freqW, powerW = ls.autopower(minimum_frequency=minf,
                                 maximum_frequency=maxf,
                                 method='slow',
                                 **kwargs)

    fig, (ax, ax1) = plt.subplots(1,
                                  2,
                                  constrained_layout=True,
                                  figsize=(6, 3),
                                  gridspec_kw=dict(width_ratios=[2, 1]))

    if frequency:
        # dm1_2_uHz = 1e6 / 86400
        # ax.plot(freqW * dm1_2_uHz, powerW)
        ax.plot(freqW, powerW)
        ax.vlines([1, 2, 4], 0, 1, color='r', alpha=0.4, zorder=-1)
        ax.vlines([1/365.25], 0, 1, color='g', alpha=0.4, zorder=-1)
        ax.vlines([1/self.mtime.ptp()], 0, 1, color='m', alpha=0.4, zorder=-1)
    else:
        ax.semilogx(1 / freqW, powerW)
        ax.vlines([1, 0.5, 0.25], 0, 1, color='r', alpha=0.4, zorder=-1)
        ax.vlines([365.25], 0, 1, color='g', alpha=0.4, zorder=-1)
        ax.vlines([self.mtime.ptp()], 0, 1, color='m', alpha=0.4, zorder=-1)

    from matplotlib.backend_bases import MouseButton
    point, = ax.plot(0, 0, 'ro')
    circle, = ax1.plot(np.cos(2 * np.pi * 1 * self.mtime),
                       np.sin(2 * np.pi * 1 * self.mtime), 'x')
    ax1.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

    def on_move(event):
        # get the x and y pixel coords
        x, y = event.x, event.y
        # if event.inaxes:
        # ax = event.inaxes  # the axes instance
        # print('data coords %f %f' % (event.xdata, event.ydata), end='\t')
        # print(ls.power(event.xdata))

    def on_click(event):
        if event.inaxes is ax and event.button is MouseButton.LEFT:
            print(event.xdata)
            point.set_data(event.xdata, ls.power(event.xdata))
            circle.set_data(np.cos(2 * np.pi * event.xdata * self.mtime),
                            np.sin(2 * np.pi * event.xdata * self.mtime))
            fig.canvas.draw()
        #     print('disconnecting callback')
        #     plt.disconnect(binding_id)


    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)

    return fig, ax


def gls_indicator(self,
                  value,
                  error,
                  label,
                  recompute=False,
                  plot=True,
                  ax=None,
                  FAP=True,
                  adjust_offsets=True,
                  frequency=False,
                  bootstrap=True,
                  HZ=False,
                  gatspy=False,
                  legend=True,
                  obs=None,
                  oversampling=20,
                  plot_data_with_offsets=False,
                  color=None,
                  line_kwargs={},
                  **kwargs):
    """
    Calculate the Lomb-Scargle periodogram of any attribute. This function can
    automatically adjust offsets (for different instruments and between ESPRESSO
    fibers, for example) while calculating the periodogram, but this is slower.
    Turn this off by setting `adjust_offsets` to False.
    """

    if self.time[self.mask].size < 3:
        print(red | 'Cannot calculate periodogram! Too few points?')
        return

    same = self._periodogram_calculated_which == value

    plow, phigh = kwargs.get('plow', None), kwargs.get('phigh', None)
    freq = self.frequency_grid(plow, phigh)
    period = 1 / freq

    try:
        value = getattr(self, value)
    except AttributeError:
        exec('out = ' + value, globals(), locals())
        value = locals()['out']

    error = getattr(self, error)

    try:
        # didn't adjust offsets before but now want to do it
        if adjust_offsets and not self.GLS['gatspy']:
            recompute = True
        # adjusted offsets before but now don't want to do it
        if not adjust_offsets and self.GLS['gatspy']:
            recompute = True
    except AttributeError:
        pass

    can_adjust_offsets = self.instruments.size > 1 or self.has_before_and_after_fibers
    if not can_adjust_offsets:
        adjust_offsets = False

    if (not self.periodogram_calculated) or recompute or (not same):
        if value.size == self.mask.size:
            # use non-masked points
            m = self.mask
            # and not those which are nan
            m &= ~np.isnan(value)
        else:
            info('different dimensions, skipping application of mask')
            m = np.full_like(value, True, dtype=bool)


        can_adjust_offsets = self.instruments.size > 1 or self.has_before_and_after_fibers
        if adjust_offsets and can_adjust_offsets:
            if self.verbose:
                info(f'Adjusting {label} offsets within periodogram')

            gatspy = True
            model = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
            if obs is None:
                obs = self.obs

            model.fit(self.time[m], value[m], error[m], filts=obs[m])
            # period, power = model.periodogram_auto(oversampling=30)
            power = model.periodogram(period)
        else:
            if gatspy:
                # if self.time.size < 50:
                model = periodic.LombScargle(fit_period=False)
                # else:
                #     model = periodic.LombScargleFast()

                model.fit(self.time[m], value[m], error[m])
                # period, power = model.periodogram_auto(oversampling=30)
                power = model.periodogram(period)

            else:
                model = LombScargle(self.time[m], value[m], error[m])
                power = model.power(1 / period)

        # save it
        self.GLS = {}
        self.GLS['model'] = model
        self.GLS['period'] = period
        self.GLS['power'] = power
        self.periodogram_calculated = True
        self._periodogram_calculated_which = value
        self.GLS['gatspy'] = gatspy
        if gatspy:
            fal = partial(false_alarm_level_gatspy, self)
            self.GLS['model'].false_alarm_level = fal

    if not self.GLS['gatspy']:
        adjust_offsets = False
        # plot_data_with_offsets = False

    if self.verbose and adjust_offsets:
        info('Adjusted means:')
        ln = self._longest_name
        offsets = self.GLS['model'].ymean_by_filt_
        instruments = self.instruments.copy().astype('U16')

        # if self.has_before_and_after_fibers:
        #     # print('(note ESPRESSO offset is between before and after fiber change)')
        #     i = np.where(instruments == 'ESPRESSO')[0][0]
        #     instruments[i] = 'ESPRESSO-post'
        #     instruments = np.insert(instruments, i, 'ESPRESSO-pre')
        #     ln += 6

        s = [
            f'  {i:{ln}s}: {off:7.4f} {self.units}'
            for i, off in zip(instruments, offsets)
        ]
        print('\n'.join(s))

    if not plot:
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        ax = ax

    if kwargs.get('show_title', True):
        ax.set_title(label, loc='left')

    kw = dict(color=color, **line_kwargs)

    if frequency:
        factor = 1 #/ 86400
        ax.plot(factor / self.GLS['period'], self.GLS['power'], **kw)
    else:
        ax.semilogx(self.GLS['period'], self.GLS['power'], **kw)

    if FAP and self.time[self.mask].size > 5:
        if bootstrap:
            if self.verbose:
                info('calculating FAP with bootstrap...')

            k = dict(method='bootstrap')
            fap01 = self.GLS['model'].false_alarm_level(0.1, **k)
            fap001 = self.GLS['model'].false_alarm_level(0.01, **k)
        else:
            fap01 = self.GLS['model'].false_alarm_level(0.1)
            fap001 = self.GLS['model'].false_alarm_level(0.01)

        fap_period = kwargs.get('fap_period', 0.98 * ax.get_xlim()[1])
        for fap, fapstr in zip((fap01, fap001), ('10%', '1%')):
            ax.axhline(fap, color='k', alpha=0.3)
            ax.text(fap_period, fap, fapstr, ha='right', va='bottom',
                    fontsize=8, alpha=0.4)

    show_planets = kwargs.get('show_planets', True)
    if show_planets and self.known_planets.P is not None:
        # legend = legend & True
        y1, y2 = ax.get_ylim()
        h = 0.1 * abs(y2 - y1)
        P = 1 / self.known_planets.P if frequency else self.known_planets.P
        ax.vlines(P,
                  ymin=y2 - h,
                  ymax=y2,
                  color='m',
                  alpha=0.6,
                  label='planets')

    show_prot = kwargs.get('show_prot', True)
    if show_prot and self.prot:
        if isinstance(self.prot, tuple):  # assume it's (prot, error)
            y1, y2 = ax.get_ylim()
            h = 0.05 * abs(y2 - y1)
            kw = dict(fmt='o', color='r', alpha=0.6, lw=2, ms=2)
            prot = 1 / self.prot[0] if frequency else self.prot[0]
            ax.errorbar(x=prot, y=y2, **kw, label=r'P$_{\rm rot}$')
            # ax.vlines(self.prot[0], ymin=y2 - h, ymax=y2, **kw)
            # ax.plot(self.prot[0], y2, 'x', **kw, label=r'P$_{\rm rot}$')

        elif not np.isnan(self.prot):
            y1, y2 = ax.get_ylim()
            h = 0.05 * abs(y2 - y1)
            kw = dict(color='r', alpha=0.6, lw=2)
            prot = 1 / self.prot if frequency else self.prot
            ax.vlines(prot, ymin=y2 - h, ymax=y2, **kw)
            ax.plot(prot, y2, 'x', **kw, label=r'P$_{\rm rot}$')

    if HZ and self.HZ is not None:
        ax.axvspan(*self.HZ, color='g', alpha=0.2, zorder=-1, label='HZ')

    xlabel = 'Frequency [days$^{-1}$]' if frequency else 'Period [days]'
    ax.set(
        xlabel=xlabel,
        ylabel='Normalised Power',
        ylim=(0, None),
        xlim=(1e-10, 1) if frequency else (1, None)
    )

    labels = [line.get_label() for line in ax.lines]
    # print(labels)
    # labels = not all([l.startswith('_') for l in labels])
    if legend and labels:
        ax.legend(ncol=10,
                  bbox_to_anchor=(1, 1.12),
                  fontsize='small',
                  handletextpad=0.3)

    add_period_axis = kwargs.get('add_period_axis', True)
    if frequency and add_period_axis:
        f2P = lambda f: 1 / (f + 1e-10)
        P2f = lambda P: 1 / (P + 1e-10)
        ax2 = ax.secondary_xaxis("top", functions=(f2P, P2f))
        ax2.minorticks_off()
        ax2.set_xticks([1, 1 / 0.5, 1 / 0.2, 1 / 0.1, 1 / 0.05])
        # ax.set_xticklabels(['0', '0.2'])
        ax.set_xlim(0, 1)
        ax2.set_xlabel('Period [days]')


    return ax


gls_fwhm = partialmethod(gls_indicator, 'fwhm', 'efwhm', 'FWHM')

gls_contrast = partialmethod(gls_indicator, 'contrast', 'econtrast', 'CCF contrast')

gls_bis = partialmethod(gls_indicator, 'bispan', 'bispan_err', 'BIS')

gls_rhk = partialmethod(gls_indicator, 'rhk', 'erhk', r"log R'$_{\rm HK}$")

gls_caindex = partialmethod(gls_indicator, 'caindex', 'caindex_err', 'Ca')
gls_naindex = partialmethod(gls_indicator, 'naindex', 'naindex_err', 'Na')
gls_haindex = partialmethod(gls_indicator, 'haindex', 'haindex_err', r'H$\alpha$')



def gls_offset(self, HZ=False, nsamples=50, _sign=True):
    # if
    # if not self.has_before_and_after_fibers:
    #     print(red | 'ERROR:', 'No points before and after offset')
    #     return

    self.gls(adjust_offsets=True, plot=False, recompute=True, plow=1.1)

    fig = plt.figure(figsize=(16, 5))#, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], figure=fig)#, height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1:])
    ax3.set(xlabel='Period [days]', ylabel='Power')

    self.plot(ax=ax1, legend=False, right_ticks=False)
    # self.plot(ax=ax2, legend=False, right_ticks=False)

    m = self.GLS['model']
    for filt, ym in zip(np.unique(m.filts), m.ymean_by_filt_):
        mask = m.filts == filt
        ax1.hlines(ym, m.t[mask].min(), m.t[mask].max(), ls='--')


    freq = 1 / self.GLS['period']
    N = freq.size
    best_offset = np.ediff1d(m.ymean_by_filt_)[0]

    # periodograms for range of offsets
    offsets = np.linspace(0, best_offset, nsamples)
    power = np.empty((nsamples, N))
    mask = m.filts != 1
    for i, of in enumerate(offsets):
        yn = m.y.copy()
        yn[mask] -= of
        power[i] = LombScargle(m.t, yn, m.dy).power(freq)

    colors = plt.cm.GnBu(np.linspace(0, 0.8, nsamples))
    for i in range(nsamples):
        ax3.semilogx(
            1 / freq,
            power[i],
            color=colors[i],
        )  # alpha=0.2)

    cmap = mpl.cm.GnBu
    norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
    cax = fig.add_axes([0.85, 0.85, 0.1, 0.05])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax,
                      orientation='horizontal',
                      label='offset [m/s]')
    cb.set_ticks([0, 0.8])
    cb.set_ticklabels(['0', f'{best_offset:.2f}'])
    # plt.colorbar()
    # handles = [
    #     Line2D([0, 1], [0, 1], color=colors[0], lw=1, label=r'offset = 0'),
    #     # Line2D([0], [0], marker='o', color='k', lw=0,
    #     #         label=r'$RV_{\rm RMS}$'),
    # ]
    # ax3.legend(handles=handles, fontsize=10, handletextpad=0, borderpad=0.2)

    if self.known_planets.P is not None:
        for p in self.known_planets.P:
            ax3.axvline(p, 0.97, 1, color='m', alpha=0.8, label='planets')

    if self.prot and not np.isnan(self.prot):
        y1, y2 = ax3.get_ylim()
        h = 0.05 * abs(y2 - y1)
        # ax3.vlines(self.prot, ymin=y2 - h, ymax=y2, color='r', alpha=0.6,
        #           lw=2)
        # ax3.plot(self.prot, y2, 'x', color='r', label=r'P$_{\rm rot}$')
        ax3.axvline(self.prot, 0.97, 1, color='r', alpha=1, lw=3,
                    label='P$_{\rm rot}$')

    if HZ and self.HZ is not None:
        ax3.axvspan(*self.HZ, color='g', alpha=0.2, zorder=-1, label='HZ')

    ax3.axhline(m.false_alarm_level(0.01), ls='--', color='k', alpha=0.2)
    ax3.text(1, 1.01 * m.false_alarm_level(0.01), 'FAP 1%')

    ax1.set_title(f'{self.star}, {self.NN} observations')
    ax3.set_title(f'Adjusted RV offset: {best_offset:.4f} m/s')
    fig.tight_layout()

    if _sign:
        fig.text(1e-3,
                 0.005,
                 'CC-BY © João Faria',
                 fontsize=8,
                 color='gray',
                 ha='left',
                 va='bottom',
                 alpha=0.5)

    return fig


def plot_gls_quantity(t, y, e, mask, obs, hl=None, setax1={}, setax2={}):
    from gatspy import periodic
    periods = np.logspace(np.log10(1), np.log10(2 * t.ptp()), 1000)
    model = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
    model.fit(t[mask], y[mask], e[mask], filts=obs[mask])
    power = model.periodogram(periods)
    model.false_alarm_level = lambda x: np.zeros_like(x)
    # return model, periods, power

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].errorbar(t[mask], y[mask], e[mask], fmt='o', ms=2)
    axs[0].plot(t[mask], model.ymean_, ls='--')

    axs[1].semilogx(periods, power)

    if hl is not None:
        for p in hl:
            axs[1].axvline(p, ls='--', color='k', alpha=0.2, zorder=-1)

    setax1.setdefault('ylabel', 'RV [m/s]')
    setax1.setdefault('xlabel', 'Time [BJD]')
    axs[0].set(**setax1)

    setax2.setdefault('ylabel', 'Power')
    setax2.setdefault('xlabel', 'Period [days]')
    axs[1].set(**setax2)


def gls_paper(self):
    with plt.style.context('fast'):
        figsize = (6, 10)

        leg = ('RV', 'FWHM', r"$\log R'_{HK}$", 'BIS', 'Na', r'H$\alpha$')
        fun = (self.gls, self.gls_fwhm, self.gls_rhk, self.gls_bis,
               self.gls_naindex, self.gls_haindex)

        fig, axs = plt.subplots(len(fun), 1, figsize=figsize,
                                constrained_layout=True)

        kw = dict(frequency=False,
                  show_planets=False,
                  show_prot=False,
                  show_title=False,
                  bootstrap=False,
                  add_period_axis=False,
                  fap_period=140)


        for i, (ax, f) in enumerate(zip(axs, fun)):
            f(ax=ax, **kw)
            ax.legend().remove()
            ax.set_title(leg[i], loc='right')
            ax.set_xlim(0.8, 100)

            kwline = dict(ls='--', alpha=0.2, lw=2, zorder=-1)
            ax.axvline(11.19, color='r', **kwline)
            ax.axvline(5.12, color='r', **kwline)
            ax.axvline(85.3, color='g', **kwline)

            # fun[i](ax=ax5, **kw)

        # for ax in axs:
        #     ax.set_xticklabels(['', '1', '5', '10', '50'])

        for ax in axs[:-1]:
            ax.set_xlabel('')
        # for ax in axs[:, 0]:
        #     ax.set_xlim(1 / 10, 1 / 12.5)
        # for ax in axs[:, 1]:
        #     ax.set_xlim(1 / 4, 1 / 6)


        # self.gls_fwhm(ax=axs[1], **kw)
        # self.gls_rhk(ax=axs[2], **kw)
        # self.gls_bis(ax=axs[3], **kw)
        # for ax in axs[:-1]:
        #     ax.set_xlabel('')
        return fig