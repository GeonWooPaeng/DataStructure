// //djikstra(다익스트라)

// const INF = Infinity;

// const arr = [
//   [0, 2, 5, 1, INF, INF],
//   [2, 0, 3, 2, INF, INF],
//   [5, 3, 0, 3, 1, 5],
//   [1, 2, 3, 0, 1, INF],
//   [INF, INF, 1, 1, 0, 2],
//   [INF, INF, 5, INF, 2, 0],
// ];

// const visited = Array.from({ length: 6 }, () => false);

// const findMinIdx = (vertex) => {
//   let minDist = INF;
//   let minIdx = 0;

//   for (let i = 0; i < arr.length; i++) {
//     if (!visited[i] && dist[i] < minDist) {
//       minDist = dist[i];
//       minIdx = i;
//     }
//   }
//   return minIdx;
// };

// const djikstra = (start) => {
//   const vertex = arr[start].slice();
//   vitied[start] = true;

//   for (let i = 0; i < arr.length; i++) {
//     const minIdx = findMinIdx(vertex);
//     visited[minIdx] = true;
//     const currentVertex = arr[minIdx];
//     for (let j = 0; j < arr.length; j++) {
//       if (visited[j]) continue;
//       if (dist[j] > dist[minIdx] + currentVertex[j]) {
//         dist[j] = dist[minIdx] + currentVertex[j];
//       }
//     }
//   }
//   return vertex;
// };

// djikstra(0);
