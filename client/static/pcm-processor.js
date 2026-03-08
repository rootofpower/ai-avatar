class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunks = [];
        this.currentChunkIndex = 0;
        this.bufferedSamples = 0;
        this.isPlaying = false;
        this.isEos = false;
        
        // 0.1 seconds threshold for snappier startup playback
        this.startThreshold = 24000 * 0.1; 
        
        this.port.onmessage = (e) => {
            if (e.data.type === 'flush') {
                this.chunks = [];
                this.currentChunkIndex = 0;
                this.bufferedSamples = 0;
                this.isPlaying = false;
                this.isEos = false;
                return;
            }
            if (e.data.type === 'eos') {
                this.isEos = true;
                return;
            }
            
            // Add Float32Array to the queue
            const chunk = e.data;
            this.chunks.push(chunk);
            this.bufferedSamples += chunk.length;
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0]; // Mono output
        
        // Either we crossed the buffer threshold, OR generation ended and we have any remaining data
        if (!this.isPlaying && (this.bufferedSamples >= this.startThreshold || (this.isEos && this.bufferedSamples > 0))) {
            this.isPlaying = true;
        }
        
        for (let i = 0; i < channel.length; i++) {
            if (this.isPlaying && this.bufferedSamples > 0) {
                const chunk = this.chunks[0];
                channel[i] = chunk[this.currentChunkIndex];
                
                this.currentChunkIndex++;
                this.bufferedSamples--;
                
                // If we finished the current chunk, discard it and move to next
                if (this.currentChunkIndex >= chunk.length) {
                    this.chunks.shift();
                    this.currentChunkIndex = 0;
                }
            } else {
                channel[i] = 0;
                if (this.isPlaying && this.bufferedSamples === 0) {
                    this.isPlaying = false;
                }
            }
        }
        return true; 
    }
}

registerProcessor('pcm-processor', PCMProcessor);
