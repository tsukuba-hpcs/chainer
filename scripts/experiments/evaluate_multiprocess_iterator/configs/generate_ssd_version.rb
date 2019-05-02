require 'json'

Dir.glob('./*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  json['train'] = '/work/imagenet/256x256/train.ssv'
  json['root'] = '/work/imagenet/256x256/train'
  puts JSON.pretty_generate(json)
  new_file_name = file.gsub('nfs', 'ssd')
  File.write(new_file_name, JSON.pretty_generate(json))
end

