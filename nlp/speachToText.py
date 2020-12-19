import speech_recognition as sr

r = sr.Recognizer()

harvard = sr.AudioFile('2020-10-17 20-12-19.wav')
with harvard as source:
    print(source.DURATION)
    secCount = 0
    while secCount < source.DURATION:
        try:
            audio = r.record(source, 10)
            secCount += 10
            text = r.recognize_google(audio)
            print(text)
            with open("transcript-10-g.txt", "a") as f:
                f.write(text+"\n")
        except Exception as e:
            print(e)
