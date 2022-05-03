let x_vals = [];    // im at 27.,28 in linear r0.2ion tf... shiffman
let y_vals = [];
let m, b;
const learningRate = 0.2;
 const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(400, 400);
  
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));

}

function draw() {
  
  tf.tidy(() => {
    if(x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });
  
  background(0);
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    point(screen_pos_x(x_vals[i]), screen_pos_y(y_vals[i]));
  }
  
  console.log(tf.memory().numTensors);
  //noLoop();
  
  const check_xs = [-0.9, 0.9];
  const check_ys = tf.tidy(() => predict(check_xs));
  //check_xs.print();
  //check_ys.print();
  
  
  //plot the predicted line
  let x1 = (map(check_xs[0], -1, 1, 0, width));
  let x2 = (map(check_xs[1], -1, 1, 0, width));
  
  // y points are in tensor so need extracting
  let lineY = check_ys.dataSync();
  check_ys.dispose();
  let y1 = (map(lineY[0], -1, 1, height, 0));
  let y2 = (map(lineY[1], -1, 1, height, 0));
  
  line(x1, y1, x2, y2);
  
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = mx + b
  //const ys = tfxs.mul(m).add(b);
  const ys = xs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  let x = mouseX;
  let y = mouseY;
  x = coord_pos_x(x);
  y = coord_pos_y(y);
  x_vals.push(x);
  y_vals.push(y);
}

function coord_pos_x(x) {
  x = map(x, 0, width, -1.0, 1.0);
  return x;
}

function coord_pos_y(y) {
  y = map(y, height, 0, -1.0, 1.0);
  return y;
}

function screen_pos_x(x) {
  x = map(x, -1.0, 1.0, 0, width);
  return x;
}

function screen_pos_y(y) {
  y = map(y, -1.0, 1.0, height, 0);
  return y;
}