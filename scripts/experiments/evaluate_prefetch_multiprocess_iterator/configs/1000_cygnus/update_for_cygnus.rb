require 'json'

Dir.glob('./*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  json['train'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all/train.ssv'
  json['root'] = '/work/NBB/serihiro/dataset/imagenet/256x256_all/train'
  json['local_storage_base'] = '/scr/local_storage_base'
  File.write(file, JSON.pretty_generate(json))
end

