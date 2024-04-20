"""Utility functions for videos, plotting and computing performance metrics."""

import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm

from . import video
from . import segmentation


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)#kreiraj video objekt

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Ovime je definiran tensor -> za svaki frame imamo matricu x/y velicine i za svaku matricu imamo 3 clana -> R,G,B kanali
    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()#ovime citamo frame po frame, ret oznacava je li frame procitan ispravno
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))
        #Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed) -> zato pretvaramo u RGB
        #vraca 3D tensor [x][y][R,G,B]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #za trenutni frame inicjlaiziraj prethodni v niz koji je stavljen na 0 sve
        #The : operator znaci bilo koji/svi -> ovo odabire sve pixela height i width dimenzije a u framue se nalaze nizovi [r,b,g] koji ce postavit za svaki pixel na ovaj nacin
        #numpy ispod haube radi ovu prijdelu vrijednosti
        #https://towardsdatascience.com/slicing-numpy-arrays-like-a-ninja-e4910670ceb0
        v[count, :, :] = frame
    #zASTO TRANSPOSE?
    #tuple of ints: i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.
    #PRVA DIMENZIJA ĆE POSTAT RGB NIZOVI, DRUGA FRAME, TREĆA VISINA ZADNJA ŠIRINA
    #DOBIT ĆEMO 3 DIMENZIJE-> SVAKA ZA JEDNU BOJAU, ONDA IMAMO FRAME DIMENZIJU KOJA JE -> ODABIREMO FRAME -> ONDA ODABIREMO X -> ONDA ODABIREMO Y KOORDINATU
    v = v.transpose((3, 0, 1, 2))

    return v


def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, _, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

                                # f, w ,h , c redoslijed
    for frame in array.transpose((1, 2, 3, 0)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        # transponiraj/zamijeni prvu i drugu dimenziju -> prva dimenzija oodgovara batch tenzoru koji sadrzi niz svih videa unutar batcha a zatim je unutar svakog videa [c,f,h,w] format
        # transponiranjem prve i druge dimenzije dobit ćemo na prvom mjestu dimnezije R,G,B kanala a nakon toga grupiranjen po videaim i frameovima i pikselima jer želimo zbrjat po kanalima
        #Pretvori u kontinuirane blokove radi bržeg zbrajanja po retcima + TO JE UJVET ZA PIRMJENU view metode
        # view mijenja oblika tenzora,  -1 = postavi na onaj broj koji će dat točan prikaz i sačuvat sve podatke
        # prvi parametar 3 kaže da želimo imati 3 retka dok će se druga dimenzija posložit bilo kako -> ona će se posložit tako da će se podaci za sve pixele za svaki frame razvuc u 1 niz
        # na taj nacin R će imat redak sa vrijednostima pixela svih frameova u batch, G će imat isto, B isto
        # velicina druge dimenzije zapravo odgovara broju svih piksela od svih frameova od svih videa u treutnom batch-u
        #kasnije ćemo zbrajat tim sutpcima i dobit vrijednosti
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1] #velicina frame-a -> druga dimenzija
        #zbrajaj po dim=1 -> zbrajaj po stupcima -> ta dimenzija će se flattat i neće je bit u rezultatu
        # zbraja po retcima tenzor x
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: np.ndarray
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def bootstrap(a, b, func, samples=10000):
    """Computes a bootstrapped confidence intervals for ``func(a, b)''.

    Args:
        a (array_like): first argument to `func`.
        b (array_like): second argument to `func`.
        func (callable): Function to compute confidence intervals for.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int, optional): Number of samples to compute.
            Defaults to 10000.

    Returns:
       A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


__all__ = ["video", "segmentation", "loadvideo", "savevideo", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]
