require 'json'

Dir.glob('./*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  json['count'] = 1000
  puts JSON.pretty_generate(json)
  File.write("./1000/#{file}", JSON.pretty_generate(json))
end

