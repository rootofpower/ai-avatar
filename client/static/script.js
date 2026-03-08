import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

const ws = new WebSocket(`ws://${location.host}/ws/chat`);
const statusEl = document.getElementById('status');
const inputEl = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const liveCanvas = document.getElementById('live-canvas');

// --- 1. Three.js & VRM Setup ---
const renderer = new THREE.WebGLRenderer({ canvas: liveCanvas, alpha: true, antialias: true });
renderer.setSize(512, 512);
renderer.setPixelRatio(window.devicePixelRatio);

const camera = new THREE.PerspectiveCamera(30.0, 1.0, 0.1, 20.0);
camera.position.set(0.0, 1.40, 0.8); // Zoomed in much closer to face, eye level

const scene = new THREE.Scene();

const light = new THREE.DirectionalLight(0xffffff, Math.PI);
light.position.set(1.0, 1.0, 1.0).normalize();
scene.add(light);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

let currentVrm = null;
let mixer = null;
const clock = new THREE.Clock();

const loader = new GLTFLoader();
loader.crossOrigin = 'anonymous';
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
    '/static/test.vrm',
    (gltf) => {
        const vrm = gltf.userData.vrm;
        if (currentVrm) {
            scene.remove(currentVrm.scene);
            VRMUtils.removeUnusedVRM(currentVrm);
        }
        currentVrm = vrm;
        scene.add(vrm.scene);
        
        vrm.scene.traverse((obj) => { obj.frustumCulled = false; });
        vrm.scene.rotation.y = 0.0;

        if (vrm.humanoid) {
            const leftArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm') || vrm.humanoid.getRawBoneNode('leftUpperArm');
            if (leftArm) leftArm.rotation.z = -Math.PI / 2.5; 
            
            const rightArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm') || vrm.humanoid.getRawBoneNode('rightUpperArm');
            if (rightArm) rightArm.rotation.z = Math.PI / 2.5;
        }

        statusEl.innerText = "Status: Connected (VRM Loaded)";
    },
    (progress) => console.log('Loading VRM...', 100.0 * (progress.loaded / progress.total), '%'),
    (error) => console.error(error)
);


// --- 2. Audio & LipSync Setup ---
let audioContext;
let analyser;
let freqData;
let avatarState = "idle"; // "idle", "thinking", "speaking"
let workletNode;

// LipSync & Animation Smoothing
let currentBlendshapes = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
let targetBlendshapes = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };

// Head and Eye tracking targets
let targetHeadRot = { x: 0, y: 0, z: 0 };
let currentHeadRot = { x: 0, y: 0, z: 0 };
let targetEyeRot = { x: 0, y: 0 };
let currentEyeRot = { x: 0, y: 0 };

async function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        
        await audioContext.audioWorklet.addModule('/static/pcm-processor.js');
        workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
        
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        analyser.smoothingTimeConstant = 0.5;
        freqData = new Uint8Array(analyser.frequencyBinCount);
        
        workletNode.connect(analyser);
        analyser.connect(audioContext.destination);
    }
}

async function playAudioChunk(blob) {
    if (!workletNode || !audioContext) return;

    // Firefox strictly suspends idle audio contexts, forcefully wake it up before pushing data
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    const arrayBuffer = await blob.arrayBuffer();
    const view = new Int16Array(arrayBuffer);
    const floatArray = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
        floatArray[i] = view[i] / 32768.0;
    }
    
    // Push raw float data to ring buffer in worklet processor
    workletNode.port.postMessage(floatArray);
}

// --- 3. Animation Loop ---
function updateLipSync() {
    if (!currentVrm) return;

    const time = clock.elapsedTime;

    // Handle states
    if (avatarState === "speaking" && analyser) {
        analyser.getByteFrequencyData(freqData);
        
        // Calculate energy in specific bands
        const getEnergy = (startHz, endHz) => {
            const nyquist = audioContext.sampleRate / 2;
            const startIndex = Math.floor(startHz / nyquist * freqData.length);
            const endIndex = Math.ceil(endHz / nyquist * freqData.length);
            let sum = 0;
            for (let i = startIndex; i < endIndex; i++) sum += freqData[i];
            return (sum / (endIndex - startIndex)) / 255.0; // Normalized 0-1
        };

        const low = getEnergy(100, 500); 
        const mid = getEnergy(500, 1500); 
        const high = getEnergy(1500, 3000); 
        const overall = getEnergy(100, 3000);
        
        targetBlendshapes.aa = 0;
        targetBlendshapes.ou = 0;
        targetBlendshapes.ih = 0;

        if (overall > 0.02) {
            targetBlendshapes.aa = Math.min(1.0, overall * 3.0);
            if (low > mid * 2.0) {
                targetBlendshapes.ou = Math.min(1.0, low * 2.0);
                targetBlendshapes.aa *= 0.3; 
            }
            if (high > mid * 0.5) {
                targetBlendshapes.ih = Math.min(1.0, high * 2.5);
                targetBlendshapes.aa *= 0.5; 
            }
        }
        
        // Keep eyes open and look at user
        currentVrm.expressionManager.setValue('blink', 0.0);
        targetHeadRot = { x: 0, y: 0, z: 0 };
        targetEyeRot = { x: 0, y: 0 };

    } else if (avatarState === "thinking") {
        targetBlendshapes.aa = 0;
        targetBlendshapes.ou = 0;
        targetBlendshapes.ih = 0;
        
        // Looks up and to the right, slightly tilts head
        targetHeadRot = { x: -0.1, y: 0.15, z: 0.05 };
        targetEyeRot = { x: -0.1, y: 0.2 };
        currentVrm.expressionManager.setValue('blink', 0.0);

    } else {
        // Idle
        targetBlendshapes.aa = 0;
        targetBlendshapes.ou = 0;
        targetBlendshapes.ih = 0;
        
        // Look straight
        targetHeadRot = { x: 0, y: 0, z: 0 };
        targetEyeRot = { x: 0, y: 0 };
        
        // Idle Blinking
        if (Math.sin(time * 3) > 0.95) {
            currentVrm.expressionManager.setValue('blink', 1.0);
        } else {
            currentVrm.expressionManager.setValue('blink', 0.0);
        }
    }
    
    // Smooth Mouth Movement (Lerp)
    const lerpFactor = 0.18;
    for (const shape of ['aa', 'ou', 'ih']) {
        currentBlendshapes[shape] += (targetBlendshapes[shape] - currentBlendshapes[shape]) * lerpFactor;
        currentVrm.expressionManager.setValue(shape, currentBlendshapes[shape]);
    }

    // Apply Head and Eye Rotations smoothly
    const headNode = currentVrm.humanoid.getNormalizedBoneNode('head');
    const leftEye = currentVrm.humanoid.getNormalizedBoneNode('leftEye');
    const rightEye = currentVrm.humanoid.getNormalizedBoneNode('rightEye');

    currentHeadRot.x += (targetHeadRot.x - currentHeadRot.x) * 0.1;
    currentHeadRot.y += (targetHeadRot.y - currentHeadRot.y) * 0.1;
    currentHeadRot.z += (targetHeadRot.z - currentHeadRot.z) * 0.1;
    
    currentEyeRot.x += (targetEyeRot.x - currentEyeRot.x) * 0.2;
    currentEyeRot.y += (targetEyeRot.y - currentEyeRot.y) * 0.2;

    if (headNode) {
        headNode.rotation.x = currentHeadRot.x;
        headNode.rotation.y = currentHeadRot.y;
        headNode.rotation.z = currentHeadRot.z;
    }
    if (leftEye) {
        leftEye.rotation.x = currentEyeRot.x;
        leftEye.rotation.y = currentEyeRot.y;
    }
    if (rightEye) {
        rightEye.rotation.x = currentEyeRot.x;
        rightEye.rotation.y = currentEyeRot.y;
    }
    
    // Subtle breathing idle animation (moves whole body slightly up/down)
    currentVrm.scene.position.y = Math.sin(time * 1.5) * 0.005;
}

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    if (currentVrm) {
        updateLipSync();
        currentVrm.update(deltaTime);
    }
    renderer.render(scene, camera);
}
animate();

// --- 4. WebSocket Events ---
ws.onopen = () => {
    statusEl.innerText = "Status: Connected (Loading VRM...)";
};

ws.onmessage = async (event) => {
    if (event.data instanceof Blob) {
        if (avatarState !== "speaking") {
            avatarState = "speaking";
            statusEl.innerText = "Status: Speaking...";
        }
        await playAudioChunk(event.data);
    } else {
        const msg = JSON.parse(event.data);
        if (msg.type === "state") {
            if (msg.value === "thinking") {
                avatarState = "thinking";
                statusEl.innerText = "Status: Thinking...";
            } else if (msg.value === "idle") {
                statusEl.innerText = "Status: Connected (Idle)";
                // Broadcast End Of Stream so worklet plays remaining buffer immediately
                if (workletNode) workletNode.port.postMessage({ type: 'eos' });
                // Add slight delay before stopping lipsync to let audio trailing finish
                setTimeout(() => { avatarState = "idle"; }, 500);
            }
        }
    }
};

sendBtn.onclick = async () => {
    await initAudio();
    const text = inputEl.value;
    if (text) {
        if (workletNode) workletNode.port.postMessage({ type: 'flush' });
        ws.send(JSON.stringify({type: 'prompt', text: text}));
        inputEl.value = '';
        statusEl.innerText = "Status: Generating...";
    }
};

inputEl.addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    event.preventDefault();
    sendBtn.click();
  }
});
