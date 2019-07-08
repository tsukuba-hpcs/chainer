require 'json'

Dir.glob('./config_ssd*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  json['train'] = '/scr/256x256_all/train.ssv'
  json['root'] = '/scr/256x256_all/train'
  File.write(file, JSON.pretty_generate(json))
end

