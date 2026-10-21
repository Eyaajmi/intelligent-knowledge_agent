import { provideHttpClient } from '@angular/common/http';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { AppComponent } from './app.component';
import { ApiService, StreamEvent } from './services/api.service';

const DOCS = { documents: [{ id: 'data/uploads/a.pdf', name: 'a.pdf', chunks: 12, pages: 3 }] };

describe('AppComponent', () => {
  let fixture: ComponentFixture<AppComponent>;
  let api: jasmine.SpyObj<ApiService>;

  beforeEach(async () => {
    api = jasmine.createSpyObj<ApiService>('ApiService', ['getDocuments', 'uploadDocument', 'deleteDocument', 'askStream', 'pageUrl']);
    api.getDocuments.and.returnValue(of(DOCS));
    api.pageUrl.and.callFake((s, p) => `http://api/documents/file?source=${s}#page=${p}`);
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideHttpClient(), { provide: ApiService, useValue: api }],
    }).compileComponents();
    fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
  });

  const text = () => (fixture.nativeElement as HTMLElement).textContent ?? '';

  it('affiche le titre et les documents indexés', () => {
    expect(text()).toContain('Intelligent Knowledge Agent');
    expect(text()).toContain('a.pdf');
    expect(text()).toContain('3 pages');
  });

  it('affiche la réponse en flux avec ses sources', async () => {
    api.askStream.and.callFake(async (_body, onEvent: (e: StreamEvent) => void) => {
      onEvent({ type: 'meta', tool: 'search', answerable: true, rewritten_question: null,
        sources: [{ source: 'data/uploads/a.pdf', name: 'a.pdf', page: 2, snippet: 'huit jours', score: 0.8 }] });
      onEvent({ type: 'token', text: 'Huit ' });
      onEvent({ type: 'token', text: 'jours.' });
      onEvent({ type: 'done' });
    });
    fixture.componentInstance.question = 'Durée maximale ?';
    await fixture.componentInstance.ask();
    fixture.detectChanges();

    expect(text()).toContain('Huit jours.');
    expect(text()).toContain('a.pdf — page 2');
    expect(fixture.componentInstance.asking).toBeFalse();
  });

  it('envoie l’historique et le document sélectionné', async () => {
    api.askStream.and.resolveTo();
    const cmp = fixture.componentInstance;
    cmp.messages = [{ role: 'user', content: 'Q1' }, { role: 'assistant', content: 'R1' }];
    cmp.selectedSource = 'data/uploads/a.pdf';
    cmp.question = 'Et ensuite ?';
    await cmp.ask();
    const body = api.askStream.calls.mostRecent().args[0];
    expect(body.source).toBe('data/uploads/a.pdf');
    expect(body.history).toEqual([{ role: 'user', content: 'Q1' }, { role: 'assistant', content: 'R1' }]);
  });

  it('montre le message d’erreur du serveur', async () => {
    api.askStream.and.rejectWith(new (await import('./services/api.service')).ApiError('Ollama est indisponible', 503));
    fixture.componentInstance.question = 'Bonjour ?';
    await fixture.componentInstance.ask();
    fixture.detectChanges();
    expect(text()).toContain('Ollama est indisponible');
  });

  it('ne supprime pas un document sans confirmation', () => {
    spyOn(window, 'confirm').and.returnValue(false);
    fixture.componentInstance.deleteDocument(DOCS.documents[0]);
    expect(api.deleteDocument).not.toHaveBeenCalled();
  });

  it('recharge les documents après suppression confirmée', () => {
    spyOn(window, 'confirm').and.returnValue(true);
    api.deleteDocument.and.returnValue(of({}));
    fixture.componentInstance.deleteDocument(DOCS.documents[0]);
    expect(api.deleteDocument).toHaveBeenCalledWith('data/uploads/a.pdf');
    expect(api.getDocuments).toHaveBeenCalledTimes(2);
  });

  it('signale l’échec du chargement des documents', () => {
    api.getDocuments.and.returnValue(throwError(() => new Error('x')));
    fixture.componentInstance.loadDocuments();
    expect(fixture.componentInstance.documentsError).not.toBe('');
  });
});
