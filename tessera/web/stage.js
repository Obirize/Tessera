// Tessera 3D stage: the finished piece on a plain wall, the build plates on a table,
// exploded view, part picking. Units are millimetres throughout.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const WALL = 0xe8e2d7;
const RED = 0xc2391f;

function noiseTexture(size, base, amp, grain = false) {
  // quiet procedural surface: plaster for the wall, oak for the table
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  const [r0, g0, b0] = base;
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let y = 0; y < size; y++) {
    const band = grain ? Math.sin(y * 0.09 + Math.sin(y * 0.013) * 6) * 0.5 + Math.sin(y * 0.41) * 0.25 : 0;
    for (let x = 0; x < size; x++) {
      const n = (rnd() - 0.5) * amp + band * amp * 1.6;
      const i = (y * size + x) * 4;
      img.data[i] = r0 + n; img.data[i + 1] = g0 + n * 0.92; img.data[i + 2] = b0 + n * 0.8; img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  return t;
}

export class Stage {
  constructor(host, { onPick, onReady } = {}) {
    this.host = host;
    this.onPick = onPick || (() => {});
    this.onReady = onReady || (() => {});
    this.view = 'art';
    this.explode = 0;
    this.showLattice = true;
    this.showPanels = false;
    this.az = -125; this.el = 36;           // one fixed, raking light from the upper left
    this.meta = null;
    this.selected = null;
    this.needs = true;

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.NoToneMapping;      // red is red and white is white: no colour grading
    host.appendChild(r.domElement);

    const s = this.scene = new THREE.Scene();
    s.background = new THREE.Color(0xe9e4db);
    this.camera = new THREE.PerspectiveCamera(26, 1, 5, 20000);
    this.camera.position.set(0, 0, 2000);

    const c = this.controls = new OrbitControls(this.camera, r.domElement);
    c.enableDamping = true; c.dampingFactor = 0.09;
    c.screenSpacePanning = true;
    c.minDistance = 120; c.maxDistance = 6000;
    c.zoomSpeed = 0.9; c.rotateSpeed = 0.7;
    c.addEventListener('change', () => (this.needs = true));

    // light: a soft gallery spot that can be swung around, plus a warm room fill
    this.hemi = new THREE.HemisphereLight(0xffffff, 0xd9d6d0, 1.35);
    s.add(this.hemi);
    const L = this.light = new THREE.SpotLight(0xffffff, 3.0, 0, Math.PI / 6.2, 0.8, 0);
    L.castShadow = true;
    L.shadow.mapSize.set(4096, 4096);
    L.shadow.bias = -0.00004;
    L.shadow.normalBias = 0.35;
    L.shadow.radius = 3;
    L.shadow.camera.near = 300; L.shadow.camera.far = 6000;
    s.add(L, L.target);

    // environment
    this.env = new THREE.Group(); s.add(this.env);
    const wallMat = new THREE.MeshStandardMaterial({ color: 0xffffff, map: noiseTexture(256, [214, 209, 200], 5),
      roughness: 0.96, metalness: 0 });
    wallMat.map.repeat.set(10, 7);
    this.wall = new THREE.Mesh(new THREE.PlaneGeometry(9000, 6000), wallMat);
    this.wall.receiveShadow = true;
    const floorMat = new THREE.MeshStandardMaterial({ color: 0xffffff, map: noiseTexture(512, [196, 180, 156], 6, true),
      roughness: 0.82, metalness: 0 });
    floorMat.map.repeat.set(3, 3);
    this.table = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), floorMat);
    this.table.rotation.x = -Math.PI / 2;
    this.table.receiveShadow = true;
    this.env.add(this.wall, this.table);

    this.art = new THREE.Group(); s.add(this.art);          // artwork in its own frame (y up, +z out)
    this.content = new THREE.Group(); this.art.add(this.content);
    this.panelLines = new THREE.Group(); this.art.add(this.panelLines);
    this.platesGroup = new THREE.Group(); s.add(this.platesGroup);

    this.loader = new GLTFLoader();
    this.ray = new THREE.Raycaster();
    this._bindPointer();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(host);
    this.size = [424, 594];
    this._layout();
    this.setLight(this.az, this.el);
    this.frame(true);
    const loop = () => {
      requestAnimationFrame(loop);
      if (this._anim) this._anim();
      if (this.controls.update()) this.needs = true;
      if (this.needs) { this.needs = false; this.renderer.render(this.scene, this.camera); }
    };
    loop();
  }

  _resize() {
    const w = this.host.clientWidth || 800, h = this.host.clientHeight || 600;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.needs = true;
  }

  // ------------------------------------------------------------------ scene content
  async loadScene(url, meta) {
    const gltf = await this.loader.loadAsync(url);
    const keepSel = this.selected && this.selected.userData.meta ? this.selected.userData.meta.code : null;
    this._clear(this.content);
    this.meta = meta;
    const byName = {};
    // GLTFLoader strips ':' from node names ("part:K1-001" -> "partK1-001")
    for (const n of meta.nodes || []) { byName[n.name] = n; byName[THREE.PropertyBinding.sanitizeNodeName(n.name)] = n; }
    const panels = {};
    for (const p of meta.panels || []) panels[p.name] = p;
    this.parts = [];
    this.byCode = {};
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const nm = byName[o.name] || byName[o.parent && o.parent.name] || null;
      o.castShadow = true; o.receiveShadow = true;
      const mat = o.material = o.material ? o.material.clone() : new THREE.MeshStandardMaterial();
      mat.roughness = 0.9; mat.metalness = 0; mat.envMapIntensity = 0;
      if (nm && nm.hex) mat.color.set(nm.hex);
      o.userData.meta = nm;
      o.userData.base = o.position.clone();
      const ex = new THREE.Vector3();
      if (nm && nm.type === 'part' && nm.bbox && meta.size_mm) {
        // every stone moves away on its own: outwards from the middle and forward by its own amount
        const b = nm.bbox, n = this.parts.length;
        const f = Math.abs(Math.sin(n * 12.9898) * 43758.5453) % 1;
        ex.set(((b[0] + b[3]) / 2 - meta.size_mm[0] / 2) * 0.26, ((b[1] + b[4]) / 2 - meta.size_mm[1] / 2) * 0.26, 10 + 26 * f);
      } else if (nm && nm.explode) ex.fromArray(nm.explode.dir).multiplyScalar(nm.explode.distance_mm || 0);
      const pn = nm && panels[nm.panel];
      if (pn && pn.explode && nm.type !== 'part') ex.add(new THREE.Vector3().fromArray(pn.explode.dir).multiplyScalar(pn.explode.distance_mm));
      o.userData.ex = ex;
      o.userData.kind = nm ? nm.type : 'part';
      if (o.userData.kind === 'part') { this.parts.push(o); if (nm) this.byCode[nm.code] = o; }
    });
    this.content.add(gltf.scene);
    this.size = meta.size_mm || this.size;
    this._buildPanelLines();
    this._layout();
    this.setExplode(this.explode);
    this.setLattice(this.showLattice);
    if (!this._framedOnce) { this.frame(true); this._framedOnce = true; }
    if (keepSel && this.byCode[keepSel]) this.select(this.byCode[keepSel], true); else this.select(null, true);
    this.needs = true;
    this.onReady();
  }

  _clear(g) {
    for (const ch of [...g.children]) {
      g.remove(ch);
      ch.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      });
    }
  }

  _buildPanelLines() {
    this._clear(this.panelLines);
    const ps = (this.meta && this.meta.panels) || [];
    const mat = new THREE.LineDashedMaterial({ color: RED, dashSize: 6, gapSize: 4, depthTest: false, transparent: true });
    for (const p of ps) {
      const [x0, y0, x1, y1] = p.bounds;
      const z = 12;
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x0, y0, z), new THREE.Vector3(x1, y0, z), new THREE.Vector3(x1, y1, z),
        new THREE.Vector3(x0, y1, z), new THREE.Vector3(x0, y0, z)]);
      const l = new THREE.Line(g, mat);
      l.computeLineDistances();
      l.renderOrder = 10;
      l.userData.panel = p;
      this.panelLines.add(l);
      const tag = this._textSprite(p.name);
      tag.position.set(x0 + 12, y1 - 10, z);
      tag.userData.panel = p;
      this.panelLines.add(tag);
    }
    this.panelLines.visible = this.showPanels;
  }

  _caption(text) {
    // plain ink caption lying on the table in front of a plate
    const c = document.createElement('canvas');
    c.width = 512; c.height = 64;
    const g = c.getContext('2d');
    g.fillStyle = '#1e1c19';
    g.font = '600 40px Bahnschrift, Segoe UI';
    g.textBaseline = 'middle';
    g.fillText(text, 4, 34);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(240, 30),
      new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false }));
    m.geometry.translate(120, -15, 0);
    m.rotation.x = -Math.PI / 2;
    m.renderOrder = 2;
    return m;
  }

  _textSprite(text) {
    const c = document.createElement('canvas');
    c.width = 128; c.height = 64;
    const g = c.getContext('2d');
    g.fillStyle = 'rgba(194,57,31,0.95)';
    g.fillRect(0, 8, 76, 46);
    g.fillStyle = '#fff';
    g.font = '600 34px Bahnschrift, Segoe UI';
    g.textBaseline = 'middle';
    g.fillText(text, 10, 32);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true }));
    s.scale.set(26, 13, 1);
    s.center.set(0, 1);
    s.renderOrder = 11;
    return s;
  }

  // ------------------------------------------------------------------ views
  setView(v) {
    this.view = v;
    this._layout();
    this.setLight(this.az, this.el);
    this.frame(true);
    this.needs = true;
  }

  _layout() {
    const [w, h] = this.size;
    const plates = this.view === 'plates';
    this.art.visible = this.wall.visible = !plates;
    this.platesGroup.visible = this.table.visible = plates;
    this.scene.background.set(plates ? 0xd9cdb9 : 0xe6e0d5);
    if (!plates) {
      this.art.rotation.set(0, 0, 0);
      this.art.position.set(-w / 2, -h / 2, 0);
      // stones are glued straight onto the wall (R6-2); with cleats/backing the wall sits behind them
      let zmin = -0.02;
      this.content.traverse((o) => {
        if (o.isMesh && o.userData.kind === 'cleat') {
          o.geometry.computeBoundingBox();
          zmin = Math.min(zmin, o.geometry.boundingBox.min.z);
        }
      });
      this.wall.position.set(0, 0, zmin - 0.05);
    }
    this.content.traverse((o) => { if (o.isMesh && o.userData.kind === 'cleat') o.visible = this.explode > 0.02; });
  }

  frame(instant = false) {
    const cam = this.camera, ctl = this.controls;
    let center, dist, dir;
    const fit = (w, h, margin) => {
      const fov = THREE.MathUtils.degToRad(cam.fov);
      const dh = (h * margin) / 2 / Math.tan(fov / 2);
      const dw = (w * margin) / 2 / Math.tan(fov / 2) / cam.aspect;
      return Math.max(dh, dw);
    };
    if (this.view === 'plates' && this.platesBox) {
      const b = this.focusBox || this.platesBox;
      center = b.getCenter(new THREE.Vector3());
      const sz = b.getSize(new THREE.Vector3());
      dist = fit(sz.x, sz.z, 1.25);
      dir = new THREE.Vector3(0, 1.0, 0.62).normalize();
    } else {
      center = new THREE.Vector3(0, 0, 0);
      dist = fit(...this.size, 1.32);
      dir = new THREE.Vector3(0, 0, 1);          // straight on; dragging turns it
    }
    const pos = center.clone().add(dir.multiplyScalar(dist));
    if (instant) {
      cam.position.copy(pos); ctl.target.copy(center); ctl.update(); this.needs = true;
      return;
    }
    const p0 = cam.position.clone(), t0 = ctl.target.clone(), start = performance.now();
    this._anim = () => {
      const k = Math.min(1, (performance.now() - start) / 450), e = 1 - Math.pow(1 - k, 3);
      cam.position.lerpVectors(p0, pos, e); ctl.target.lerpVectors(t0, center, e);
      this.needs = true;
      if (k >= 1) this._anim = null;
    };
  }

  // ------------------------------------------------------------------ light
  setLight(az, el) {
    this.az = az; this.el = el;
    const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(Math.max(8, Math.min(85, el)));
    // az: direction to the light in picture coordinates (y down); -130 = upper left
    const local = new THREE.Vector3(Math.cos(a) * Math.cos(e), -Math.sin(a) * Math.cos(e), Math.sin(e));
    let dir = local.clone();
    let target = new THREE.Vector3(0, 0, 0);
    if (this.view === 'plates') dir = new THREE.Vector3(local.x, local.z, -local.y);
    if (this.view === 'plates' && this.platesBox) target = this.platesBox.getCenter(new THREE.Vector3());
    const span = this.view === 'plates' && this.platesBox ? this.platesBox.getSize(new THREE.Vector3()).length()
      : Math.hypot(...this.size);
    const d = Math.max(1600, span * 2.2);
    this.light.position.copy(target).add(dir.multiplyScalar(d));
    this.light.target.position.copy(target);
    this.light.angle = Math.min(Math.PI / 3.2, Math.atan((span * 0.85) / d) + 0.05);
    this.light.shadow.camera.far = d * 2;
    this.light.shadow.camera.near = d * 0.5;
    this.light.shadow.camera.updateProjectionMatrix();
    this.needs = true;
  }

  // ------------------------------------------------------------------ explode / toggles
  setExplode(t) {
    this.explode = t;
    const k = t * t * (3 - 2 * t);
    this.content.traverse((o) => {
      if (!o.isMesh || !o.userData.base) return;
      o.position.copy(o.userData.base).addScaledVector(o.userData.ex, k);
      if (o === this.selected) o.position.z += 4;
    });
    this.panelLines.position.z = 30 * k;
    this.content.traverse((o) => { if (o.isMesh && o.userData.kind === 'cleat') o.visible = t > 0.02; });
    this.needs = true;
  }

  // lifting parts towards a camera that looks straight at them shows nothing: turn to 3/4 first
  obliqueIfFrontal() {
    if (this.view === 'plates' || this._anim) return;
    const cam = this.camera, ctl = this.controls;
    const off = cam.position.clone().sub(ctl.target);
    const dist = off.length();
    if (off.clone().normalize().z < 0.93) return;
    const pos = ctl.target.clone().add(new THREE.Vector3(0.62, 0.22, 0.75).normalize().multiplyScalar(dist * 0.92));
    const p0 = cam.position.clone(), start = performance.now();
    this._anim = () => {
      const k = Math.min(1, (performance.now() - start) / 700), e = 1 - Math.pow(1 - k, 3);
      cam.position.lerpVectors(p0, pos, e);
      this.needs = true;
      if (k >= 1) this._anim = null;
    };
  }

  setLattice(on) {
    this.showLattice = on;
    this.content.traverse((o) => { if (o.isMesh && o.userData.kind === 'backing') o.visible = on; });
    this.needs = true;
  }

  setPanels(on) {
    this.showPanels = on;
    this.panelLines.visible = on;
    this.needs = true;
  }

  // ------------------------------------------------------------------ picking
  _bindPointer() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    el.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      if (moved < 4 && this.view !== 'plates') this._pick(e);
      down = null;
    });
  }

  _pick(e) {
    if (!this.parts || !this.parts.length) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(p, this.camera);
    const hit = this.ray.intersectObjects(this.parts, false)[0];
    this.select(hit ? hit.object : null);
  }

  select(obj, silent = false) {
    if (this.selected) {
      const m = this.selected.material;
      m.emissive.setHex(0x000000);
      if (this.selected.userData.edges) { this.selected.remove(this.selected.userData.edges); this.selected.userData.edges.geometry.dispose(); this.selected.userData.edges = null; }
    }
    this.selected = obj;
    if (obj) {
      obj.material.emissive = new THREE.Color(RED);
      obj.material.emissiveIntensity = 0.28;
      const eg = new THREE.EdgesGeometry(obj.geometry, 50);
      const lines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: RED }));
      obj.add(lines);
      obj.userData.edges = lines;
    }
    this.setExplode(this.explode);
    if (!silent) this.onPick(obj ? obj.userData.meta : null);
  }

  // ------------------------------------------------------------------ plates
  async loadPlates(url, meta) {
    const gltf = await this.loader.loadAsync(url);
    this._clear(this.platesGroup);
    this.platesMeta = meta;
    const g = gltf.scene;
    g.rotation.x = -Math.PI / 2;        // bed normal +z -> world up
    g.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = !o.name.startsWith('bed');
      o.receiveShadow = true;
      if (!o.material) return;
      if (o.name.startsWith('bed')) {   // a light build plate, so black parts read on it
        o.material = o.material.clone();
        o.material.color.setHex(0xa39a8a); o.material.roughness = 0.62; o.material.metalness = 0;
      } else {
        o.material = o.material.clone();
        o.material.roughness = 0.9; o.material.metalness = 0;
        const q = /^stone:?(.+)@bed(\d+)$/.exec(o.name) || [];
        const pl = (meta.plates || [])[+q[2] - 1], st = this.byCode && this.byCode[q[1]];
        const hex = (pl && pl.hex) || (st && st.userData.meta && st.userData.meta.hex);
        if (hex) o.material.color.set(hex);
      }
    });
    this.platesGroup.add(g);
    g.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(g);
    const c = box.getCenter(new THREE.Vector3());
    g.position.set(-c.x, -box.min.y + 0.6, -c.z);
    g.updateMatrixWorld(true);
    this.platesBox = new THREE.Box3().setFromObject(g);
    this.plateBoxes = (meta.plates || []).map((p) => {
      const [ox, oy] = p.origin_mm, P = p.size_mm[0];
      const b = new THREE.Box3(new THREE.Vector3(ox, oy, -3), new THREE.Vector3(ox + P, oy + P, 10));
      return b.applyMatrix4(g.matrixWorld);
    });
    (meta.plates || []).forEach((p) => {
      const tag = this._caption(p.title);
      tag.position.copy(new THREE.Vector3(p.origin_mm[0] + 2, p.origin_mm[1] - 6, 0).applyMatrix4(g.matrixWorld));
      tag.userData.caption = true;
      this.platesGroup.add(tag);
    });
    this.focusBox = null;
    if (this.view === 'plates') { this.setLight(this.az, this.el); this.frame(true); }
    this.needs = true;
  }

  focusPlate(i) {
    this.focusBox = i == null || !this.plateBoxes ? null : this.plateBoxes[i];
    for (const ch of this.platesGroup.children) if (ch.userData.caption) ch.visible = !this.focusBox;
    this.frame(false);
  }

  snapshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL('image/png'); }
}
