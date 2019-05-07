require 'json'

Dir.glob('./*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  json['count'] = 1000
  puts JSON.pretty_generate(json)
  new_file_name = file.gsub('.json', '_1000.json')
  File.write(new_file_name, JSON.pretty_generate(json))
end

