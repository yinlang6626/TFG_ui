// static/js/modules/voice-recorder.js
export class VoiceRecorder {
    constructor(options = {}) {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.onRecordingStart = options.onRecordingStart;
        this.onRecordingStop = options.onRecordingStop;
    }
    
    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                if (this.onRecordingStop) {
                    this.onRecordingStop(audioBlob);
                }
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            if (this.onRecordingStart) {
                this.onRecordingStart();
            }
            
            return true;
        } catch (error) {
            console.error('录音启动失败:', error);
            return false;
        }
    }
    
    stop() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
        }
    }
}