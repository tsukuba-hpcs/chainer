require 'json'

CONFIG_BASE = 'resnet50_config/prefetch_multiprocess_iterator'

ITERATORS = ['prefetch_multiprocess']
LOADERJOBS = [1,2,4,8,16]
PREFETCHJOBS = [1,2,4,8,16]
N_PREFETCH = [100, 500, 1000, 5000, 10000]
BATCH = [32, 64]
MAX_PROCESS = 24

BASE = {
  '_train' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/train.ssv',
  '_val' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/val.ssv',
  'root' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/train',
  'mean' => '/home/serizawa/python36_test/scripts/ImagenetConverterForChainer/mean_imagenet_all.npy',
  'local_storage_base' => '/work/imagenet/local_storage_base',
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
          config['out'] = "/home/serizawa/python36_test/prefetch_iterator/chainer/examples/imagenet/resnet50_results/prefetch_multiprocess_iterator/#{batch}_#{n_prefetch}_#{loaderjob}_#{prefetchjob}"
          file_name = "#{batch}_#{n_prefetch}_#{loaderjob}_#{prefetchjob}.json"
          File.write("#{CONFIG_BASE}/#{file_name}", JSON.pretty_generate(config))
        end
      end
    end
  end
end

