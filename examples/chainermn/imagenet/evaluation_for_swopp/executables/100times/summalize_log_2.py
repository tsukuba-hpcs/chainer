import argparse

'''
StandardUpdater.update_core(): 42 0 0.1696016788482666 4.6253204345703125e-05 0.010028839111328125 0.15950703620910645
_MultiNodeOptimizer.update():  4 0.15234875679016113 0.04620814323425293 0.0955960750579834 0.005529642105102539
StandardUpdater.update_core(): 4 0 0.16959190368652344 3.504753112792969e-05 0.010117530822753906 0.15941953659057617
_MultiNodeOptimizer.update():  56 0.15245819091796875 0.04602766036987305 0.09530878067016602 0.0056345462799072266
StandardUpdater.update_core(): 56 0 0.16959118843078613 3.2901763916015625e-05 0.00978398323059082 0.15975451469421387
_MultiNodeOptimizer.update():  61 0.1522507667541504 0.04540419578552246 0.09563231468200684 0.005408763885498047
StandardUpdater.update_core(): 61 0 0.16959810256958008 3.314018249511719e-05 0.009618043899536133 0.1599271297454834
_MultiNodeOptimizer.update():  39 0.1523759365081787 0.04628324508666992 0.09453392028808594 0.005536794662475586
StandardUpdater.update_core(): 39 0 0.1696169376373291 3.0279159545898438e-05 0.009747743606567383 0.15981698036193848
_MultiNodeOptimizer.update():  52 0.1520688533782959 0.045995473861694336 0.09529638290405273 0.005251646041870117
StandardUpdater.update_core(): 52 0 0.16957759857177734 4.506111145019531e-05 0.009686708450317383 0.15982365608215332
_MultiNodeOptimizer.update():  45 0.15305876731872559 0.04907798767089844 0.09522843360900879 0.006216526031494141
StandardUpdater.update_core(): 45 0 0.16958832740783691 8.20159912109375e-05 0.009885549545288086 0.15959906578063965
_MultiNodeOptimizer.update():  36 0.15257525444030762 0.04482865333557129 0.09558391571044922 0.005756855010986328
StandardUpdater.update_core(): 36 0 0.16965198516845703 3.457069396972656e-05 0.009782791137695312 0.15979862213134766
_MultiNodeOptimizer.update():  55 0.15214800834655762 0.046133995056152344 0.09482169151306152 0.0053098201751708984
StandardUpdater.update_core(): 55 0 0.16959643363952637 3.0994415283203125e-05 0.00933694839477539 0.16020631790161133
'''

def main(log):
    load_minibatch = []
    forward = []
    allreduce = []
    backward = []
    update_total = []
    converter = []
    new_epoch = []
    optimizer_update = []

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
                update_total.append(float(splitted[3]))
                load_minibatch.append(float(splitted[4]))
                converter.append(float(splitted[5]))
                optimizer_update.append(float(splitted[6]))
                new_epoch.append(float(splitted[7]))

    print(len(load_minibatch), len(forward), len(allreduce), len(backward))

    print(
                f'{sum(load_minibatch)},' +
                f'{sum(converter)},' + 
                f'{sum(forward)},' + 
                f'{sum(allreduce)},' + 
                f'{sum(backward)},' + 
                f'{sum(new_epoch)},' +
                f'{sum(load_minibatch) + sum(converter) +  sum(forward) +  sum(allreduce) + sum(backward) + sum(new_epoch)}/{sum(update_total)},' +
                f'{sum(optimizer_update)} / {sum(update_total)},' + 
                f'{sum(optimizer_update) + sum(load_minibatch)}  / {sum(update_total)}'
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", "-l", type=str, required=True)

    args = parser.parse_args()

    base = '/work/NBB/serihiro/src/chainer/examples/chainermn/imagenet/evaluation_for_swopp/executables/100times'
    main(base + '/' +  args.log)

