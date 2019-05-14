require 'json'

CONFIG_BASE = 'resnet50_config/multiprocess_iterator'

ITERATORS = ['multiprocess']
LOADERJOBS = [1,2,4,8,16]
N_PREFETCH = [100, 500, 1000, 5000, 10000]
BATCH = [32, 64]

BASE = {
  '_train' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/train.ssv',
  '_val' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/val.ssv',
  'root' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/train',
  'mean' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/mean_imagenet_all.npy',
  'gpu' => 0,
  'arch' => 'resnet50',
  'epoch' => 1
}

ITERATORS.each do |iterator|
  LOADERJOBS.each do |loaderjob|
    N_PREFETCH.each do |n_prefetch|
      BATCH.each do |batch|
        config = BASE.dup
        config['iterator'] = iterator
        config['loaderjob'] = loaderjob
        config['n_prefetch'] = n_prefetch
        config['batchsize'] = batch
        config['val_batchsize'] = batch
        config['out'] = "/home/serizawa/python36_test/prefetch_iterator/chainer/examples/imagenet/resnet50_results/multiprocess_iterator/nfs/#{batch}_#{n_prefetch}_#{loaderjob}"
        file_name = "nfs/#{batch}_#{n_prefetch}_#{loaderjob}.json"
        File.write("#{CONFIG_BASE}/#{file_name}", JSON.pretty_generate(config))
      end
    end
  end
end

