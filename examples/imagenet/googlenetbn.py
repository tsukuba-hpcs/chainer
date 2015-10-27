import chainer
import chainer.functions as F
import chainer.links as L


class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__(
            conv1=F.Convolution2D(3, 64, 7, stride=2, pad=3, nobias=True),
            norm1=F.BatchNormalization(64),
            conv2=F.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=F.BatchNormalization(192),
            inc3a=F.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=F.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=F.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=F.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=F.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=F.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=F.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=F.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=F.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=F.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out=F.Linear(1024, 1000),

            conva=F.Convolution2D(576, 128, 1, nobias=True),
            norma=F.BatchNormalization(128),
            lina=F.Linear(2048, 1024, nobias=True),
            norma2=F.BatchNormalization(1024),
            outa=F.Linear(1024, 1000),

            convb=F.Convolution2D(576, 128, 1, nobias=True),
            normb=F.BatchNormalization(128),
            linb=F.Linear(2048, 1024, nobias=True),
            normb2=F.BatchNormalization(1024),
            outb=F.Linear(1024, 1000),
        )
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.inc3a.train = value
        self.inc3b.train = value
        self.inc3c.train = value
        self.inc4a.train = value
        self.inc4b.train = value
        self.inc4c.train = value
        self.inc4d.train = value
        self.inc4e.train = value
        self.inc5a.train = value
        self.inc5b.train = value

    def __call__(self, x, t):
        test = not self.train

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a), test=test))
        a = F.relu(self.norma2(self.lina(a), test=test))
        a = self.outa(a)
        self.loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b), test=test))
        b = F.relu(self.normb2(self.linb(b), test=test))
        b = self.outb(b)
        self.loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (self.loss1 + self.loss2) + self.loss3
        self.accuracy = F.accuracy(h, t)
        return self.loss
