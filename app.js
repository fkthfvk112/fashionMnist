//서버
const express = require('express');

//page로부터 정보 받기(정보를 라우터에 저장해 접근 가능하도록 해줌)
const bodyParser = require('body-parser');

//tesnsorflowjs
const tf = require('@tensorflow/tfjs');

//경로처리
const path = require('path');

//multipart처리(이미지 받기) 
const multer = require('multer');

//이미지 수정 모듈
const sharp = require('sharp');

const storage = multer.memoryStorage();
const upload = multer({ storage: storage});


const app = express();

//템플릿 엔진 설정
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// body-parser 설정
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());



app.get('/', (req, res)=>{
  res.render('pageOne')
})


//POST 처리
app.post('/predict', upload.single('image'), async(req, res) => {
  try {
    //이미지 버퍼 받아오기
    const buffer = req.file.buffer;
    // console.log('파일', req.file)
    // console.log('버퍼', buffer)
    // console.log('사이즈', buffer.length)

    //리사이즈
    const resizedImage = await sharp(buffer)
    .resize({ width: 28, height: 28, fit: 'cover', position:'center' })
    .greyscale()
    .toBuffer();
    console.log('resized buffer ', resizedImage)

    //28*28 버퍼->배열로 변환

    //0으로 채운 28 by 28 2차원 배열
    const pixels = new Uint8Array(resizedImage);
    console.log('픽셀', pixels)
    let arr = new Array(28);
    for(let i = 0; i < 28; i++){
      arr[i] = new Array(28).fill(0);
    }

    const width = Math.floor(Math.sqrt(pixels.length));
    let index = 0;
    for(let i = 0; i < width; i++){
      for(let j = 0; j<width; j++){
        arr[i][j] = pixels[index];
        index++;
      }
    }
    console.log('이미지', arr);

    //배열 to 2d 텐서 to 3d 텐서
    const tensor2d = tf.tensor2d(arr);
    const tensor3d = tensor2d.expandDims(-1);
    console.log('텐서 ', tensor3d);

    // 모델 로드 및 예측

    const modelPath = path.resolve(__dirname, 'model', 'model.json');//변환된 모델 삽입
    const model = await tf.loadLayersModel(modelPath);
    console.log('모델 ', model);
    const prediction = model.predict(tensor3d);
    const predictedClass = prediction.argMax(1).dataSync()[0];

    // 예측 결과를 클라이언트에게 전송
    res.send(predictedClass);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error occurred during prediction');
  }
});

// 서버 실행
app.listen(8080, () => {
  console.log('Server is running on port 8080');
});
