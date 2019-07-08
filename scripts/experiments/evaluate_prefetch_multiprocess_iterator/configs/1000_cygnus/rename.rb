Dir.glob('./*.json').each do |file|
    splitted = file.split('_')
    new_file_name = [splitted[0], splitted[1], splitted[3], splitted[4], '1000'].join('_') + '.json'
    puts(new_file_name)
    File.rename(file, new_file_name)
end

