import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { ApiError, ApiService, detailToMessage, StreamEvent } from './api.service';

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

describe('ApiService', () => {
  let service: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  it('récupère les documents', () => {
    service.getDocuments().subscribe((r) => expect(r.documents.length).toBe(1));
    const req = http.expectOne((r) => r.url.endsWith('/documents'));
    expect(req.request.method).toBe('GET');
    req.flush({ documents: [{ id: 'x', name: 'x.pdf', chunks: 1, pages: 1 }] });
  });

  it('supprime un document par son identifiant (en paramètre, pas dans le chemin)', () => {
    service.deleteDocument('data/uploads/a.pdf').subscribe();
    const req = http.expectOne((r) => r.url.endsWith('/documents'));
    expect(req.request.method).toBe('DELETE');
    expect(req.request.params.get('source')).toBe('data/uploads/a.pdf');
    req.flush({});
  });

  it('construit un lien vers la page du PDF', () => {
    const url = service.pageUrl('data/uploads/mon doc.pdf', 16);
    expect(url).toContain('/documents/file?source=data%2Fuploads%2Fmon%20doc.pdf');
    expect(url.endsWith('#page=16')).toBeTrue();
  });

  it('reconstitue les événements SSE même coupés entre deux paquets', async () => {
    const e1 = JSON.stringify({ type: 'token', text: 'Bon' });
    const e2 = JSON.stringify({ type: 'done' });
    const payload = `data: ${e1}\n\ndata: ${e2}\n\n`;
    const cut = 9; // coupe au milieu du premier événement
    spyOn(window, 'fetch').and.resolveTo(sseResponse([payload.slice(0, cut), payload.slice(cut)]));

    const events: StreamEvent[] = [];
    await service.askStream({ question: 'q', source: null, history: [] }, (e) => events.push(e));
    expect(events).toEqual([{ type: 'token', text: 'Bon' }, { type: 'done' }]);
  });

  it('lève une ApiError avec le détail du serveur', async () => {
    spyOn(window, 'fetch').and.resolveTo(
      new Response(JSON.stringify({ detail: 'Le document sélectionné n’existe plus.' }), { status: 404 }),
    );
    await expectAsync(service.askStream({ question: 'q', source: 'x', history: [] }, () => {})).toBeRejectedWith(
      jasmine.objectContaining({ message: 'Le document sélectionné n’existe plus.', status: 404 }),
    );
  });

  it('convertit les erreurs de validation FastAPI en texte', () => {
    expect(detailToMessage([{ msg: 'Value error, vide' }], 'x')).toBe('Value error, vide');
    expect(detailToMessage(undefined, 'repli')).toBe('repli');
    expect(new ApiError('m', 400).status).toBe(400);
  });
});
