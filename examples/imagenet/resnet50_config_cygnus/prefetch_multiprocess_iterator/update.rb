require 'json'

Dir.glob('./*.json').each do |file|
  text = File.read(file)
  json = JSON.parse(text)
  json['_train'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv'
  json['_val'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all/val.ssv'
  json['root'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all'
  json['mean'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all/mean.npy'
  json['out'] = "/work/NBB/serihiro/src/chainer/examples/imagenet/resnet50_results/prefetch_multiprocess_iterator#{file.split('.')[-2]}"
  json['local_storage_base'] = '/scr/local_storage_base'
  File.write(file, JSON.pretty_generate(json))
end
