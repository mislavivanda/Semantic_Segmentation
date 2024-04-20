"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas

import numpy as np
import skimage.draw
import torchvision
import echonet


#PYTHON INHERITA OVU DATASET KLASU OD torchvisiona->dolje je koristit kod super() poziva kako bi pristupio njenim funkcijama
#Base Class For making datasets which are compatible with torchvision.
#It is necessary to override the __getitem__ and __len__ method.
class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    #OVO JE FUNKCIJA KONSTRUKTORA U PYTHONU(ISTA KAO KLJUČNA RIJEC CONSTRUCTOR U JS/C++)
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        if root is None:
            root = echonet.config.DATA_DIR

        #POZIV KONSTRUKTORA PARENT KLASe torchvision.datasets
        super().__init__(root, target_transform=target_transform)

        #SELF PREDSTAVLJA INSTNACU OBJEKTA KOJI SE KREIRA KOD POZIVA OVOG KONSTRUKTORA
        #self OBJET ĆE SE ZAPRAVO VRATIT NA MJESTU GDJE SE POZIVA KONSTRUKTOR KLASE
        #NA TAJ NAČIN MU POSTAVLJAMO PARAMETRE KOJE SMO DOBILI POZIVOM echonet.datasets.Echo() funkcije
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                #uzmi samo one koji imaju Split isti kao i prolsijeđeni, inače uzmi sve ako je stavljen All
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()#imena stupaca u obliku liste
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()#values atribut je tensor/niz koji sadrzi svaki redak u obliku niza za svaki file

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces -> OZNAČENI LEFT VENTRICULI U X/Y SUSTAVU SLIKE -> DEFINIRAJU BOX UNUTAR KOJEG SE NALAZI LEFT VENTRICULE
            #COLLECTIONS.DEFAULTDICT JE POSEBAN EXTENDANI TIP PODATAKA U PYTHONU
            #DICTIONARY JE KAO MAPA -> FORMAT key-value pairs
            self.frames = collections.defaultdict(list)#DICTIONARY WITH VALUES OF LIST -> vrijednost svakog eleenta dictionary je lista
            self.trace = collections.defaultdict(_defaultdict_of_lists) #OVO ĆE BIT DICTIONARY ČIJI ĆE ELEMENTI BIT DICTIONARY
            #keyevi za gornje dictionarye će bit:
            #1)za self.frame to ce biti [filename] -> za svaki video(fillename) ćemo imat listu frameova tog videa
            #2) za self.trace to ce bit [filename][frame] buduci da imamo za odredeni video i svaki njegov frame određene x,y koordinate onda ih sprememoa na razini frame za svaki video
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                #X1,Y1,X2,Y2 BI MOGLE BIT VKOORDINATE KOSIH PRAVACA KOJI SU NASLAGANI JEDNA IZNAD DRUGOG I KOJI KAD SE ISCRTAJU DEFINIRAJU POVRSINU VENTRICULA
                #PUNIMO IH ZA SVAKI FRAME OD VIDEA U NIZ U REDOLSIJEDU X1,Y1,X2,Y2 I KASNIJE IH PRETVORIMO U NIZ KOJI CE IC PO OVOM REDSOLIJEDU
                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))#FORMATE/REDSOLIJED KOORDINATA LEFT VENTRICULA
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])#TENSOR SA NIZOVIMA CDULJINE 4 KOJI DEFINIRAJU DUZINU KOJA DEFINIRA VELICINU VENTRICULA

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

            #NAKON OVOGO KORAKA IMAMO RAZLIČITE PODATKE KOJI SE TIČU CIJELOG DATASETA + 2 LISTE -> LISTA SVIH FRAMEOVA ZA SVAKI VIDEO I LISTA X,Y KOORDINATA ZA SVAKI FRAME OD SVAKOG VIDEA

    def __getitem__(self, index):
        #index= INDEX videa u direktoriju odakle se dohvaćaju podaci
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        #format c,f,h,w
        video = echonet.utils.loadvideo(video).astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            #.SHAPE VRAĆA TUPLE() KOJI SADRZI DIMENZIJE VIDEA -> ZNAMO DA JE TO (3,BROJFRAMEOVA,SIRINA,VISINA)
            # n je zapravo ukupan broj pixela u cijelom videu
            n = video.shape[1] * video.shape[2] * video.shape[3]
            #prvi parametar randoma je ovi n -> onece se pretborit u niz ko da pise np.arrange(n) -> to će bit niz [0,1,...,n]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            #ind predstavlja niz brojeva -> geneira se vise brojeva i to definira 2. parametar
            #numpy dozovljava da na mjesto indexiranja stavimo niz -> time odabiremo ele,ente na tim pozicijama
            #npr: niz [1, 2, 3, 4, 5]
            # arr[[0,1,2]] -> ispis [1,2,3]
            #npr-> imamo 20 frameova i svaki ima 200 pixela -> ukupno 20*200 pixela
            #dobili random broj 201 -> treba obojat prvi pixel od drugog framea 
            f = ind % video.shape[1]#svedemo se na indeksaciju vrameova tako da radimo % -> dobit ćemo indekse nekih frameova
            ind //= video.shape[1]#ovime dobivamo broj pixela po frameu
            i = ind % video.shape[2]#odaberemo neki redak
            ind //= video.shape[2]#ovime dobivamo broj pixela po retku -> ne triba radit % jer smo vec u tom rasponu -> samp proslijedi vrijdnost
            j = ind
            #za svaku boju za određeni frame i pozicije x,y postavimo vrijednosti 0 -> CRNA
            video[:, f, i, j] = 0

        # Apply normalization -> svedi na raspon [0-1]
        #normalna varijabla -> z=(x-aritmeticka)/devijacija
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        #channels, frames, height, width od videa
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            #Sampling period for taking a clip from the video (i.e. every ``period''-th frame is take
            length = f // self.period#self period oznacava svako koji frame uzet
            #length=broj frameova po clipu
        else:
            # Take specified number of frames-> ako je proslijedeno tako
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        #broj frameova manji od broja frameova koje trebamo uzet
        #length * slefp eriod oznacuju da cemo npr zako ocmo uzimat 16 frameova s periodom 5 duljina videa mora bit minimalno 80 framoeva jer ćemo ić od poečtka
        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        #U start se nalaze pocetne pozicije klipova unutar videa
        if self.clips == "all":
            #svi clipovi ce ukupno imat length frameova
            #svaki clip ce imat length/broj clipova frameova
            #clip radimo tako da idemo od pocetka i uzimamo svako period frame
            #clip=razbit video na jos manje djelove/clipove
            # Take all possible clips of desired length
            #f=broj frameova videa
            #length oznacava Number of frames to clip from video. If ``None'', longest possible clip is returned
            #ako je length=f/period(length=none-> uzmi najduzi clip)onda ce vrijednost biti period -> izlazni niz ce biti [0,1,2,...period]-> broj klipova ce bit jednak broju perioda
            #miocemo uzimat frameove 0, period, 2*period, ..., (length-1)*period
            #f-(length-1)*period oznacava preostale frameove
            #to znaci da nam je presotalo jos toliko frameova -> mozemo pocet od pocetnog framea ili od bilo kojeg framea u tom offsetu i zadrzat ćemo se unutar videa-> zato kiristimo arrange
            #dobivamo evenly spaced niz kojo oznacava pocetke clipova
            #najkrajnji pocetak clippa moze biti takav da nakon njega ostane sa zadanim periodom length broj frameova
            #vratit će indekse od 0 do tog zadnjeg indeksa
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            #uzmi onoliko clipova koliko je specificirano u self.clips
            #izaberi pocetnu random poziciju od mogucih za svaki od clipova
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        #TARGET= STA ZELIMO SVE SPREMIT/UZET
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":#INDEX LARGETS FRAMEA
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":#INDEX SMALLEST FRAMEA
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                #ZADNJI FRAME=DIJASTOLA
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                #PRVI FRAME=SISTOLA
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    #ZADNJI FRAME GLEDAMO ZA ZADANI FILENAME U self.trace dictionaryu
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                #koordinate
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                #U X1 SUS SVE X KOORDINATE OD X1 TOCKA ZA ZADANI FRAME, ISTA LOGIKA I U X2,Y1 I Y2 -> NIZOVI
                #FLIP=Reverse the order of elements in an array along the given axis
                #preskoći prve 2 točke jer one predstavljaju glavnu središnju liniju od dna prema vrhu klijetke a ona nam nije bitna za iscrtavanje
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                #zašto flip-> jer crtamo od dna prema vrhu -> prvo pocinjemo s iscrtavanjem tocaka(x1,y1) i spajamo ih ravnin linijaa, time smo dosli do vrha ventircula -> sad se trebamo spustit dole -> uzimamo (x2,y2) koordinate od najgornjih linija i idemo prema dolje
                #pRVI PARAMETAR=Row coordinates of vertices of polygo
                #dRUGI PARAMETAR=Column coordinates of vertices of polygon
                #TOCKA KOJA SE CRTA JE ZAPRAVO PSARIVANJE ELEMENATA NA ISTIM INDEKSIMA U ROW I COLUMN ARRAYIMA
                #kad iscrtamo tocke sve on se povezuju s linijama -> time smo dobili prikaz linijama koi pokriva podrucje ventricula
                #FUNKCIJA VRAČA BITMAPU U KOJOJ SU SA 1 OZNAČENI PIXELI KOJI PRIPADAJU POLIGONU -> U NAŠEM SLUČAJU VENTIRCULU
                #ZADNJI PARAMETAR=IMAGE SHAPE=VELICINA SLIKE=> POSTAVIMO DA ZELIMO IZLAZNU MATICU VLIČINE VIDEA JER ĆEMO OVO INTEGRIRAT U VIDEO
                #VRACA r,c koji predstavljaju koordinate pixela koji su iscrtani(vrijdnost 1, svima ostalima 0)
                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                #maska velicine videa
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1#postavi pixele koji trebaju bit iscrtani na 1
                #SPRENA TU MASKU I KASNIJE JE KORISTIMO KOD RACUNANJA PARAMETARA EPOHE I PREKLAPANJA REZULTATA MODELA I TRAIN PODATAKA
                target.append(mask)#pohrani masku
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            #pretvori u tuple
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        #drugi agument ce odabrat one frameovoe koje trebamo uzet, start ce definirat offset 
        #tuple je formata (clip1, clip2, clip3, ...)
        #svaki clip je formata c,f,h,w
                                #uzima frame-ove od pffseta/starta pa sve do duljine videa specificrane u length, duljina koja treba bit je 16 a period 2 start = 10 -> uzimamo 10, 12, 14, 16, .... frameove u jedan klip
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            #ako je clips=1 tada imamo samo 1 video i uzimamo njega iz tuplea, jedini i prvi član tuplea
            #uzmi prvi video -> cijeli video ce bit jednak jednom clipu
            video = video[0]
        else:
            #joinaj u jedan niz -> imat cemo unutar video objekta clipove jedna iza drugog
            video = np.stack(video)

        if self.pad is not None:
            #kada zelimo siru sliku od rezolucije videa onda ostatke paddamo s crnom bojom odnosno srednjom bojom
            # Add padding of zeros (mean color of videos)-> ako je normalna varijabla z=0-> x=aritmeticka sredina
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            #dodjaemo padding na sirinu i visinu framea simetricno s obe strane -> povecaj dimenzije za te brojeve
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            #na mjestima di nece bit 0 postavi vriiednosti boje od videa
            # self.pad:-self.pad -> indeksi self.pad, self.pad+1, ... DULJINA - (self.pad) će biti postavljeni na vrijednosti videa
            # POČETNI PIKSELI S LIJEVE/DESNE/GORNJE/DONJE SLIKE ĆE BITI POSTAVLJENI NA 0 A U SREDINI ĆE BITI PIKSELI VIDEA
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)#geneiraj 2 random broja(zadnji parametar 2) u rasponu od 0 do 2*self.pad(sirina paddinga kojeg dodajemo)
            #
            video = temp[:, :, i:(i + h), j:(j + w)]

        #All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least 224 pixels
        #Povratna vrijednot ove metode je niz clipova/batcheva(od 1 clana ili vise) i svaki ima 
        return video, target

    def __len__(self):
        #broj videa=broj filenameova
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
