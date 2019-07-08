require 'json'

n_list = [250, 500, 750, 1000]

Dir.glob('./*.json').each do |file|
  raw_text = File.read(file)
  json = JSON.parse(raw_text)
  n_list.each do |n|
    json['n_prefetch'] = n
    splited = file.split('_')
    splited[2] = n.to_s
    new_file_name = splited.join('_')
    File.write(new_file_name, JSON.pretty_generate(json))
  end
end

