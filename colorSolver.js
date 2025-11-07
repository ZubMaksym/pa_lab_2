'use strict'
// Usage: node colorSolver.js --nodes=20 --runs=20 --beamK=10

const os = require('os');
const { argv } = require('process');

const args = {};
argv.slice(2).forEach(a => {
    const m = a.match(/^--([^=]+)=?(.*)$/);
    if (m) args[m[1]] = m[2] === '' ? true : (m[2] || true);
});

const NODES = parseInt(args.nodes || 20, 10);
const RUNS = parseInt(args.runs || 20, 10);
const BEAM_K = parseInt(args.beamK || 10, 10);
const MAX_ITER_BEAM = parseInt(args.maxIterBeam || 5000, 10);
const COLORS = parseInt(args.colors || 4, 10);

const TIME_LIMIT_MS = 30 * 60 * 1000; // 30 min
const MEM_LIMIT_BYTES = 1 * 1024 * 1024 * 1024; // 1 GB

function randInt(a, b) {
    return Math.floor(Math.random() * (b - a + 1)) + a;
}

function generateMap(n, extraEdgeProb = 0.12) {
    const adj = Array.from({ length: n }, () => new Set());

    for (let v = 1; v < n; ++v) {
        const u = randInt(0, v - 1);
        adj[u].add(v); adj[v].add(u);
    }

    for (let u = 0; u < n; ++u) {
        for (let v = u + 1; v < n; ++v) {
            if (Math.random() < extraEdgeProb) {
                adj[u].add(v); adj[v].add(u);
            }
        }
    }
    return adj.map(s => Array.from(s));
}

function degrees(adj) {
    return adj.map(nei => nei.length);
}

function heuristicDGR(assign, adj) {
    let score = 0;
    const deg = degrees(adj);
    for (let u = 0; u < adj.length; ++u) {
        for (const v of adj[u]) {
            if (v > u && assign[u] === assign[v]) score += (deg[u] + deg[v]);
        }
    }
    return score;
}

function heuristicMY(assign, adj) {
    let conflicts = 0;
    const conflictCountPerNode = new Array(adj.length).fill(0);
    for (let u = 0; u < adj.length; ++u) {
        for (const v of adj[u]) {
            if (v > u && assign[u] === assign[v]) {
                conflicts++;
                conflictCountPerNode[u]++; conflictCountPerNode[v]++;
            }
        }
    }
    let penalty = 0;
    for (const c of conflictCountPerNode) {
        if (c > 1) penalty += (c - 1) * 0.5;
    }
    return conflicts + penalty;
}

function solveBacktrackingWithHeuristic(adj, colors = COLORS, heuristic = 'DGR', timeLimit = TIME_LIMIT_MS, memLimit = MEM_LIMIT_BYTES) {
    const n = adj.length;
    const assign = new Array(n).fill(-1);
    let generated = 0;
    let deadEnds = 0;
    let maxStack = 0;
    const start = Date.now();
    let stopped = false;

    function checkLimits() {
        if (Date.now() - start > timeLimit) { stopped = true; return true; }
        if (process.memoryUsage().heapUsed > memLimit) { stopped = true; return true; }
        return false;
    }

    function validAssign(v, c) {
        for (const u of adj[v]) {
            if (assign[u] === c) return false;
        }
        return true;
    }

    function vertexScore(v) {
        const deg = adj[v].length;
        if (heuristic === 'DGR') return deg;
        if (heuristic === 'MY') {
            let conflicts = 0;
            for (const u of adj[v]) if (assign[u] !== -1 && assign[u] === assign[v]) conflicts++;
            return conflicts;
        }
        return 0;
    }

    function selectNextVertex() {
        let candidates = [];
        for (let v = 0; v < n; ++v) if (assign[v] === -1) candidates.push(v);
        if (candidates.length === 0) return -1;
        candidates.sort((a, b) => vertexScore(b) - vertexScore(a));
        return candidates[0];
    }

    let steps = 0;
    let found = false;

    function backtrack() {
        if (stopped) return false;
        const v = selectNextVertex();
        if (v === -1) { found = true; return true; }
        maxStack = Math.max(maxStack, n - assign.filter(x => x === -1).length + 1);
        steps++;

        for (let c = 0; c < colors; ++c) {
            generated++;
            if (validAssign(v, c)) {
                assign[v] = c;
                if (backtrack()) return true;
                assign[v] = -1;
            }
            if (checkLimits()) return false;
        }
        deadEnds++;
        return false;
    }

    backtrack();

    return {
        found,
        assign: found ? assign.slice() : null,
        generatedStates: generated,
        deadEnds,
        steps,
        maxMemoryStates: maxStack,
        timeMs: Date.now() - start,
        stopped
    };
}

function beamSearch(adj, k = BEAM_K, colors = COLORS, heuristicName = 'DGR', maxIter = MAX_ITER_BEAM, timeLimit = TIME_LIMIT_MS, memLimit = MEM_LIMIT_BYTES, opts = {}) {
    const n = adj.length;
    const start = Date.now();
    let stopped = false;

    function checkLimits() {
        if (Date.now() - start > timeLimit) { stopped = true; return true; }
        const mem = process.memoryUsage().heapUsed;
        if (mem > memLimit) { stopped = true; return true; }
        return false;
    }

    function evalState(s) {
        if (heuristicName === 'DGR') return heuristicDGR(s, adj);
        return heuristicMY(s, adj);
    }

    // === Змінено: тільки одне початкове розфарбування ===
    const initialState = opts.initialColoring || Array.from({ length: n }, () => randInt(0, colors - 1));
    const beams = [{ s: initialState.slice(), score: evalState(initialState) }];

    let generated = 0;
    let iter = 0;
    const seenHashes = new Set();
    function hashState(s) { return s.join(','); }

    while (iter < maxIter) {
        if (checkLimits()) break;
        iter++;

        // Перевірка, чи знайдено рішення
        for (const b of beams) if (b.score === 0) {
            return { found: true, assign: b.s.slice(), generatedStates: generated, steps: iter, beamSize: beams.length, timeMs: Date.now() - start, stopped: false };
        }

        const neighbors = [];
        // Для всіх станів в beam генеруємо сусідів
        for (const b of beams) {
            const s = b.s;
            for (let v = 0; v < n; ++v) {
                const original = s[v];
                for (let c = 0; c < colors; ++c) {
                    if (c === original) continue;
                    const ns = s.slice();
                    ns[v] = c;
                    generated++;
                    neighbors.push({ s: ns, score: evalState(ns) });
                }
            }
        }

        if (neighbors.length === 0) break;

        neighbors.sort((a, b) => a.score - b.score);
        const newBeams = [];
        let i = 0;
        while (newBeams.length < k && i < neighbors.length) {
            const h = hashState(neighbors[i].s);
            if (!seenHashes.has(h)) {
                newBeams.push(neighbors[i]);
                seenHashes.add(h);
            }
            i++;
        }

        // Замінюємо старі beam на нові
        beams.length = 0;
        beams.push(...newBeams);
    }

    return {
        found: false,
        assign: null,
        generatedStates: generated,
        steps: iter,
        beamSize: k,
        timeMs: Date.now() - start,
        stopped
    };
}

function singleExperiment(mapAdj, alg, options = {}) {
    if (alg === 'BCTR') {
        return solveBacktrackingWithHeuristic(mapAdj, options.colors || COLORS, options.heuristic || 'DGR', options.timeLimit || TIME_LIMIT_MS, options.memLimit || MEM_LIMIT_BYTES);
    }
    else if (alg === 'BEAM') {
        return beamSearch(mapAdj, options.k || BEAM_K, options.colors || COLORS, options.heuristic || 'DGR', options.maxIter || MAX_ITER_BEAM, options.timeLimit || TIME_LIMIT_MS, options.memLimit || MEM_LIMIT_BYTES, options);
    } else {
        throw new Error('Unknown alg');
    }
}

function runSeries(alg, mapGenerator, runs = RUNS, opts = {}) {
    const results = [];
    let mapAdj = opts.fixedMap ? opts.fixedMap : mapGenerator();

    for (let i = 0; i < runs; ++i) {
        const res = singleExperiment(mapAdj, alg, opts);
        results.push({ mapAdj, res });
        if (res.stopped) {
            console.warn(`[WARN] Run ${i} for ${alg} was stopped due to time/mem limits.`);
        }
    }
    return results;
}

function summarizeResults(resList) {
    const n = resList.length;
    let foundCount = 0;
    let stepsSum = 0;
    let deadEndsSum = 0;
    let generatedSum = 0;
    let memSum = 0;
    let timeSum = 0;
    for (const r of resList) {
        const s = r.res;
        if (s.found) foundCount++;
        stepsSum += (s.steps || 0);
        deadEndsSum += (s.deadEnds || 0);
        generatedSum += (s.generatedStates || 0);
        memSum += (s.maxMemoryStates || s.beamSize || 0);
        timeSum += s.timeMs || 0;
    }
    return {
        runs: n,
        foundCount,
        foundPct: (foundCount / n * 100).toFixed(1),
        avgSteps: (stepsSum / n).toFixed(2),
        avgDeadEnds: (deadEndsSum / n).toFixed(2),
        avgGenerated: (generatedSum / n).toFixed(2),
        avgMemoryStates: (memSum / n).toFixed(2),
        avgTimeMs: (timeSum / n).toFixed(0)
    };
}


async function main() {
    console.log('COLOR solver. Nodes:', NODES, 'Runs:', RUNS, 'Colors:', COLORS, 'BeamK:', BEAM_K);
    const sharedMap = generateMap(NODES, 0.12);
    const initialColoring = Array.from({ length: NODES }, () => randInt(0, COLORS - 1));
    console.log(initialColoring);

    const ENABLE = {
        BCTR_DGR: true,
        BCTR_MY: true,
        BEAM_DGR: true,
        BEAM_MY: true
    };

    let bctrDGRResults = [];
    let bctrMYResults = [];
    let beamDGRResults = [];
    let beamMYResults = [];

    if (ENABLE.BCTR_DGR) {
        console.log('\nRunning Backtracking (BCTR) with DGR');
        bctrDGRResults = runSeries('BCTR', generateMap, RUNS, { heuristic: 'DGR', fixedMap: sharedMap });
        console.log('BCTR (DGR) summary:', summarizeResults(bctrDGRResults));
    }

    if (ENABLE.BCTR_MY) {
        console.log('\nRunning Backtracking (BCTR) with MY');
        bctrMYResults = runSeries('BCTR', generateMap, RUNS, { heuristic: 'MY', fixedMap: sharedMap });
        console.log('BCTR (MY) summary:', summarizeResults(bctrMYResults));
    }

    if (ENABLE.BEAM_DGR) {
        console.log('\nRunning Beam search (BEAM) with DGR');
        beamDGRResults = runSeries('BEAM', generateMap, RUNS, {
            heuristic: 'DGR',
            k: BEAM_K,
            maxIter: MAX_ITER_BEAM,
            fixedMap: sharedMap,
            initialColoring
        });
        console.log('BEAM (DGR) summary:', summarizeResults(beamDGRResults));
    }

    if (ENABLE.BEAM_MY) {
        console.log('\nRunning Beam search (BEAM) with MY');
        beamMYResults = runSeries('BEAM', generateMap, RUNS, {
            heuristic: 'MY',
            k: BEAM_K,
            maxIter: MAX_ITER_BEAM,
            fixedMap: sharedMap,
            initialColoring
        });
        console.log('BEAM (MY) summary:', summarizeResults(beamMYResults));
    }

    const allResults = [bctrDGRResults, bctrMYResults, beamDGRResults, beamMYResults].flat();
    const sample = allResults.find(x => x.res.found);
    if (sample) {
        console.log('\nExample solution (one of runs):');
        console.log('Map degrees:', degrees(sample.mapAdj));
        if (sample.res.assign) console.log('Assignment:', sample.res.assign.join(' '));

        const fs = require('fs');
        console.log('\nGenerating DOT visualization...');

        let dot = 'graph G {\n  layout=neato;\n  overlap=false;\n  node [shape=circle style=filled fontsize=10];\n';
        for (let i = 0; i < sample.res.assign.length; i++) {
            const colorMap = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231'];
            const color = colorMap[sample.res.assign[i]] || '#aaaaaa';
            dot += `  ${i} [label="${i}", fillcolor="${color}"];\n`;
        }
        for (let i = 0; i < sample.mapAdj.length; i++) {
            for (const j of sample.mapAdj[i]) {
                if (j > i) dot += `  ${i} -- ${j};\n`;
            }
        }
        dot += '}\n';

        fs.writeFileSync('graph.dot', dot);
        console.log('DOT-file saved as graph.dot');
        console.log('Open https://dreampuf.github.io/GraphvizOnline/.');
    }

    console.log('\nDONE.');
}

main().catch(e => {
    console.error('Fatal error:', e);
});
