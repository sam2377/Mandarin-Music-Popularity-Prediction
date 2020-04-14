from pydub import AudioSegment

files_path = './Song/'
file_name = ''

startMin = 0
startSec = 0

endMin = 0
endSec = 30

# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

if __name__ == "__main__":
    for i in range(1,1924) :
        file_name = str(i).zfill(5)
        print(file_name)
        song = AudioSegment.from_mp3(files_path+file_name + '.mp3')
        extract = song[startTime:endTime]
        extract.export('./Ex-Song/' + file_name + '-ex.mp3', format="mp3")