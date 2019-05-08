require 'json'

CONFIG_BASE = 'resnet50_config/prefetch_multiprocess_iterator'

ITERATORS = ['prefetch_multiprocess']
LOADERJOBS = [1,2,4,8,16]
PREFETCHJOBS = [1,2,4,8,16]
N_PREFETCH = [100, 500, 1000, 5000, 10000]
BATCH = [32, 64]
MAX_PROCESS = 24

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
    PREFETCHJOBS.each do |prefetchjob|
      N_PREFETCH.each do |n_prefetch|
        BATCH.each do |batch|
          if loaderjob + prefetchjob > MAX_PROCESS
            next
          end
          config = BASE.dup
          config['iterator'] = iterator
          config['loaderjob'] = loaderjob
          config['prefetchjob'] = prefetchjob
          config['n_prefetch'] = n_prefetch
          config['batchsize'] = batch
          config['val_batchsize'] = batch
          file_name = "#{batch}_#{n_prefetch}_#{loaderjob}_#{prefetchjob}.json"
          File.write("#{CONFIG_BASE}/#{file_name}", JSON.pretty_generate(config))
        end
      end
    end
  end
end

