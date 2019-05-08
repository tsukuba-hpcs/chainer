require 'json'

CONFIG_BASE = 'resnet50_config/multiprocess_iterator'

ITERATORS = ['multiprocess']
LOADERJOBS = [1,2,4,8,16]
N_PREFETCH = [100, 500, 1000, 5000, 10000]
BATCH = [32, 64]

BASE = {
  '_train' => '/work/imagenet/256x256_all/train.ssv',
  '_val' => '/work/imagenet/256x256_all/val.ssv',
  'root' => '/work/imagenet/256x256_all',
  'mean' => '/work/imagenet/256x256_all/mean.npy',
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
        file_name = "#{batch}_#{n_prefetch}_#{loaderjob}.json"
        File.write("#{CONFIG_BASE}/#{file_name}", JSON.pretty_generate(config))
      end
    end
  end
end

