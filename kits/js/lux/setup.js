const fs = require("fs");
const readline = require("readline");
const setup = () => {

  // Prepare to read input
  const rl = readline.createInterface({
    input: process.stdin,
    output: null,
  });

  let buffer = [];
  let currentResolve;
  const makePromise = function () {
    return new Promise((resolve) => {
      currentResolve = resolve;
    });
  };
  // on each line, push line to buffer
  rl.on('line', (line) => {
    buffer.push(line);
    currentResolve();
    currentPromise = makePromise();
  });

  // The current promise for retrieving the next line
  let currentPromise = makePromise();

  // with await, we pause process until there is input
  const getLine = async () => {
    return new Promise(async (resolve) => {
      while (buffer.length === 0) {
        // pause while buffer is empty, continue if new line read
        await currentPromise;
      }
      // once buffer is not empty, resolve the most recent line in stdin, and remove it
      resolve(buffer.shift());
    });
  };
  return getLine;
}
module.exports = { setup }
