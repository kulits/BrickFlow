import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { LD } from './viewer.js';

export function manipulate(v, items, { onStart, onChange, onEnd, bound, slide }) {
  const canvas = v.renderer.domElement, coarse = matchMedia('(pointer: coarse)').matches;
  const lift = new TransformControls(v.camera, canvas), turn = new TransformControls(v.camera, canvas), proxy = new THREE.Object3D();
  lift.setMode('translate'); lift.setSize(0.55); lift.setSpace('world'); lift.showX = lift.showZ = false;
  turn.setMode('rotate'); turn.setSize(coarse ? 1.5 : 1.6); turn.setSpace('world');
  for (const rings of [turn._gizmo.picker.rotate, turn._gizmo.gizmo.rotate]) rings.remove(...rings.children.filter((o) => o.name === 'XYZE'));
  v.overlayScene.add(lift.getHelper(), turn.getHelper(), proxy);

  let sel = null, offset = new THREE.Matrix4(), body = null, down = null;
  const busy = v.touchGrab = () => lift.dragging || turn.dragging || !!body;
  function attach(item) {
    sel = item;
    const M = new THREE.Matrix4().multiplyMatrices(LD, new THREE.Matrix4().set(...item.get()));
    proxy.position.copy(new THREE.Vector3(...item.com).applyMatrix4(M));
    proxy.quaternion.setFromRotationMatrix(M);
    proxy.updateMatrixWorld(true);
    offset = proxy.matrixWorld.clone().invert().multiply(M);
    if (!coarse) lift.attach(proxy);
    turn.attach(proxy);
    v.request();
  }
  function detach() { sel = null; lift.detach(); turn.detach(); v.request(); }
  function commit() {
    const d = bound && proxy.position.clone().sub(bound.center);
    if (d && d.length() > bound.radius) proxy.position.copy(bound.center).add(d.setLength(bound.radius));
    proxy.updateMatrixWorld(true);
    sel.set(Float64Array.from(new THREE.Matrix4().multiplyMatrices(LD, proxy.matrixWorld.clone().multiply(offset)).transpose().elements));
    onChange();
    v.request();
  }
  for (const [tc, other] of [[lift, turn], [turn, lift]]) {
    const hold = (on) => () => { other.enabled = v.controls.enabled = !on; (on ? onStart : onEnd)(); };
    tc.addEventListener('change', () => v.request());
    tc.addEventListener('objectChange', () => sel && commit());
    tc.addEventListener('mouseDown', hold(true));
    tc.addEventListener('mouseUp', hold(false));
  }

  const ray = new THREE.Raycaster(), plane = new THREE.Plane(), hit = new THREE.Vector3();
  function aim(e) {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, 1 - (e.clientY - r.top) / r.height * 2), v.camera);
  }
  function pick(e) {
    aim(e);
    const meshes = items.flatMap((item) => item.group.children.filter((o) => o.isMesh).map((o) => Object.assign(o, { item })));
    const h = ray.intersectObjects(meshes, false)[0];
    return h && { item: h.object.item, point: h.point };
  }
  canvas.addEventListener('pointerdown', (e) => {
    down = { x: e.clientX, y: e.clientY, handle: lift.dragging || turn.dragging };
    const h = !down.handle && e.button === 0 && pick(e);
    if (!h || (e.pointerType === 'touch' && sel !== h.item)) return;
    if (sel !== h.item) attach(h.item);
    const n = v.camera.getWorldDirection(new THREE.Vector3());
    if (slide !== 'view' && Math.abs(n.y) > 0.25) n.set(0, 1, 0);
    plane.setFromNormalAndCoplanarPoint(n, h.point);
    body = { start: h.point.clone(), from: proxy.position.clone(), id: e.pointerId };
    v.controls.enabled = false;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    onStart();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (body?.id === e.pointerId) {
      aim(e);
      if (ray.ray.intersectPlane(plane, hit)) { proxy.position.copy(body.from).add(hit.clone().sub(body.start)); commit(); }
    } else if (!lift.dragging && !turn.dragging) canvas.style.cursor = lift.axis || turn.axis ? 'pointer' : pick(e) ? 'grab' : 'default';
  });
  const end = (e) => {
    const click = down && !down.handle && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4;
    if (body) {
      body = null;
      v.controls.enabled = true;
      canvas.style.cursor = 'grab';
      attach(sel);
      onEnd();
    } else if (click && !lift.axis && !turn.axis) {
      const h = pick(e);
      h ? attach(h.item) : detach();
    }
    down = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  addEventListener('keydown', (e) => e.key === 'Escape' && sel && detach());
  return { follow: () => sel && !busy() && attach(sel), detach };
}
