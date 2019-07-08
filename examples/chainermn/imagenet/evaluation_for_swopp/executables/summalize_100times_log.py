import argparse

'''
StandardUpdater.update_core(): 28 0 0.2001953125 4.7206878662109375e-05 0.1904757022857666
_MultiNodeOptimizer.update():  20 0.15269947052001953 0.047293663024902344 0.09524059295654297 0.005277156829833984
StandardUpdater.update_core(): 20 0 0.17048907279968262 4.982948303222656e-05 0.16092252731323242
_MultiNodeOptimizer.update():  44 0.15256404876708984 0.046010732650756836 0.09488415718078613 0.005146980285644531
StandardUpdater.update_core(): 44 0 0.18971562385559082 5.793571472167969e-05 0.18023180961608887
_MultiNodeOptimizer.update():  48 0.15263605117797852 0.04570317268371582 0.09474754333496094 0.00521540641784668
StandardUpdater.update_core(): 48 0 0.20402312278747559 5.078315734863281e-05 0.19459056854248047
_MultiNodeOptimizer.update():  56 0.15265107154846191 0.04626584053039551 0.09534239768981934 0.005230426788330078
StandardUpdater.update_core(): 56 0 0.20850801467895508 4.267692565917969e-05 0.19913315773010254
_MultiNodeOptimizer.update():  16 0.1526486873626709 0.045224666595458984 0.0954594612121582 0.005223989486694336
StandardUpdater.update_core(): 16 0 0.25662779808044434 3.147125244140625e-05 0.2471153736114502
_MultiNodeOptimizer.update():  41 0.1527717113494873 0.04551053047180176 0.09463310241699219 0.005352973937988281
StandardUpdater.update_core(): 41 0 0.18110156059265137 4.696846008300781e-05 0.17158031463623047
_MultiNodeOptimizer.update():  36 0.15274548530578613 0.04656481742858887 0.09610819816589355 0.005328178405761719
StandardUpdater.update_core(): 36 0 0.19870471954345703 4.458427429199219e-05 0.1894514560699463
_MultiNodeOptimizer.update():  52 0.15266680717468262 0.04579806327819824 0.09511065483093262 0.005250453948974609
StandardUpdater.update_core(): 52 0 0.20912861824035645 5.245208740234375e-05 0.19974303245544434
_MultiNodeOptimizer.update():  60 0.1528775691986084 0.04552745819091797 0.09546136856079102 0.005460500717163086
StandardUpdater.update_core(): 60 0 0.20083236694335938 4.9114227294921875e-05 0.19137048721313477
_MultiNodeOptimizer.update():  1 0.15206241607666016 0.04617929458618164 0.0963294506072998 0.005213022232055664
StandardUpdater.update_core(): 1 0 0.2625770568847656 3.1948089599609375e-05 0.2529008388519287
_MultiNodeOptimizer.update():  2 0.15200471878051758 0.0446467399597168 0.09644269943237305 0.005151510238647461
StandardUpdater.update_core(): 2 0 0.2626173496246338 3.3855438232421875e-05 0.2533705234527588
_MultiNodeOptimizer.update():  3 0.15201354026794434 0.04508090019226074 0.09647846221923828 0.005171537399291992
StandardUpdater.update_core(): 3 0 0.26259613037109375 2.8848648071289062e-05 0.25298166275024414
_MultiNodeOptimizer.update():  0 0.1520686149597168 0.04500007629394531 0.09624719619750977 0.005220890045166016
StandardUpdater.update_core(): 0 0 0.20849847793579102 4.4345855712890625e-05 0.19908976554870605

>>> 'StandardUpdater.update_core(): 0 0 0.20849847793579102 4.4345855712890625e-05 0.19908976554870605'.split(' ')
['StandardUpdater.update_core():', '0', '0', '0.20849847793579102', '4.4345855712890625e-05', '0.19908976554870605']
'''

def main(log, iteration_epoch, process, epoch):
    load_minibatch = []
    forward = []
    allreduce = []
    backward = []
    update_total = []

    with open(log, "r") as f:
        for l in f:
            l = l.strip()
            if l.startswith("_MultiNodeOptimizer.update():"):
                splitted = l.split(' ')
                forward.append(float(splitted[4]))
                allreduce.append(float(splitted[5]))
                backward.append(float(splitted[6]))
                continue

            if l.startswith("StandardUpdater.update_core():"):
                splitted = l.split(' ')
                load_minibatch.append(float(splitted[3]))
                update_total.append(float(splitted[2]))

    print(len(load_minibatch), len(forward), len(allreduce), len(backward))

    i = 0
    for e in range(1, epoch + 1):
        t = e * iteration_epoch * process
        print(
                f'{sum(load_minibatch[i:t])},' +
                f'{sum(forward[i:t])},' + 
                f'{sum(allreduce[i:t])},' + 
                f'{sum(backward[i:t])},' + 
                f'{sum(load_minibatch[i:t]) +  sum(forward[i:t]) +  sum(allreduce[i:t]) + sum(backward[i:t])}/{sum(update_total[i:t])}'
        )

        i = t



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", "-l", type=str, required=True)
    parser.add_argument("--iteration_epoch", "-i", type=int, default=616)
    parser.add_argument("--process", "-p", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=10)

    args = parser.parse_args()

    base = '/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp/executables'
    main(base + '/' +  args.log, args.iteration_epoch, args.process, args.epoch)

