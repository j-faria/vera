import os
import contextlib
import inspect

try:
    import colorful as cl
    cl.use_style('solarized')
    blue = cl.bold_blue
    red = cl.red
    redb = cl.bold_red
    yel = cl.yellow
    yelb = cl.bold_yellow
    green = cl.green
except ImportError:
    class color:
        def __or__(self, other):
            return other
    blue = red = redb = yel = yelb = green = color()
    # raise ImportError('Please, pip install colorful')


def info(msg: str, show_info: bool = True, **kwargs):
    if show_info:
        print(blue | 'INFO:', msg, **kwargs)
    else:
        print(blue | '    :', msg, **kwargs)


def warning(msg: str, show_warn: bool = True, **kwargs):
    if show_warn:
        print(yelb | 'WARN:', msg, **kwargs)
    else:
        print(yelb | '    :', msg, **kwargs)


def error(msg: str, show_error: bool = True, **kwargs):
    if show_error:
        print(red | 'ERROR:', msg, **kwargs)
    else:
        print(red | '     :', msg, **kwargs)


@contextlib.contextmanager
def chdir(dir: str):
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


try:
    import rich
    rich_available = True
except ImportError:
    rich_available = False


def print_system_info(system):
    s = system
    if rich_available:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.table import Column, Table

        console = Console()
        console.print(s.star, style='bold')
        console.print('RVs from', s.instruments)

        table = Table(show_header=True, header_style="magenta")
        cols = ('V', 'ST', 'Teff (K)', 'Mass (Msun)')
        vals = s.vmag, s.spectral_type, s.teff, s.stellar_mass
        for col in cols:
            table.add_column(col, justify='right')
        table.add_row(*[str(v) for v in vals])

        console.print(table)
    else:
        print(s.star, 'RVs from ', s.instruments)


def get_artist_location(artist, ax, fig):
    # bbox1 = leg.get_bbox_to_anchor()
    try:
        bbox1 = artist.get_tightbbox(fig.canvas.get_renderer())  # mpl 3
    except AttributeError:
        try:
            bbox1 = artist.get_bbox(fig.canvas.get_renderer())  # mpl 2.2
        except AttributeError:
            bbox1 = artist.get_window_extent(
                fig.canvas.get_renderer())  # mpl 2.2
    bbox2 = bbox1.transformed(ax.transData.inverted())
    # print(artist, '\n', bbox1, bbox2)
    return bbox2.x0, bbox2.x1, bbox2.y0, bbox2.y1


def get_function_call_str(frame, top=True):
    fname = inspect.getframeinfo(frame).function
    # argspec = inspect.getfullargspec(frame) # doesn't work
    args_str = get_args_str(*inspect.getargvalues(frame))
    if top:
        top = 's = RV'
    else:
        top = 's'
    return f'{top}.{fname}({args_str})'


def get_args_str(args, varargs, varkw, vals):
    pairs = []
    for arg in args:
        if arg in ('cls', 'self'):
            continue
        if isinstance(vals[arg], str):
            pairs.append(f'{arg}="{vals[arg]}"')
        else:
            pairs.append(f'{arg}={vals[arg]}')

    if varargs is not None:
        for arg in vals[varargs]:
            if isinstance(vals[varargs][arg], str):
                pairs.append(f'{arg}="{vals[varargs][arg]}"')
            else:
                pairs.append(f'{arg}={vals[varargs][arg]}')

    if varkw is not None:
        for arg in vals[varkw]:
            if isinstance(vals[varkw][arg], str):
                pairs.append(f'{arg}="{vals[varkw][arg]}"')
            else:
                pairs.append(f'{arg}={vals[varkw][arg]}')

    return ', '.join(pairs)


def split_rdb_file_first_column(file, value, new_name, old_name=None):
    from numpy import loadtxt
    col = loadtxt(file, skiprows=2, usecols=(0,))
    mask = col > value

    head = 2
    tail = mask.sum()
    if tail == 0:
        print(f'No observations after {value} in file {file}')
    else:
        cmd = f'(head -n{head} && tail -n{tail}) < {file} > {new_name}'
        os.system(cmd)

    if old_name is not None:
        head = 2 + (~mask).sum()
        cmd = f'(head -n{head}) < {file} > {old_name}'
        os.system(cmd)


def find_data_file(file):
    here = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(here, '..', 'data', file)
    if not os.path.exists(data_file):
        data_file = os.path.join(here, 'data', file)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'Cannot find "{file}" in data')
    return data_file


def styleit(func):
    from matplotlib.pyplot import style
    here = os.path.dirname(__file__)

    def wrapper(*args, **kwargs):
        with style.context([os.path.join(here, 'cute_style.mplstyle')]):
            func(*args, **kwargs)

    return wrapper
