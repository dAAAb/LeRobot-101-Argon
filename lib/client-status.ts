/** Browser-only transport/status helpers. Tests use injected fetch/read mocks. */
export class ControllerRequestError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ControllerRequestError';
    this.status = status;
  }
}

type Fetcher = typeof fetch;
type Token = { current: string };
async function boundedJson<T>(
  path: string,
  init: RequestInit,
  fetcher: Fetcher,
  timeoutMs: number,
): Promise<T> {
  const abort = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const operation = (async () => {
    const response = await fetcher(path, {
      ...init,
      signal: abort.signal,
      cache: 'no-store',
    });
    if (!response.ok) {
      let detail = '連線失敗，請再試一次';
      try {
        const data = (await response.json()) as { detail?: unknown };
        if (typeof data.detail === 'string') detail = data.detail;
      } catch {}
      throw new ControllerRequestError(response.status, detail);
    }
    return (await response.json()) as T;
  })();
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      // Reject first so callers receive the uncertainty message, not a generic
      // AbortError. A POST timeout never means the controller rejected the job.
      reject(
        new ControllerRequestError(
          0,
          init.method === 'POST'
            ? '請求逾時，暫時無法確認是否已受理。請先查看工作狀態。'
            : '連線逾時，暫時無法更新工作狀態。',
        ),
      );
      abort.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([operation, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

export async function controllerRequest<T>(
  path: string,
  body: unknown,
  token: Token,
  fetcher: Fetcher = fetch,
  timeoutMs = 6000,
): Promise<T> {
  async function session() {
    const data = await boundedJson<{ token: string }>(
      '/api/session',
      {},
      fetcher,
      timeoutMs,
    );
    if (typeof data.token !== 'string' || !data.token)
      throw new ControllerRequestError(0, '本機控制器尚未就緒');
    token.current = data.token;
  }
  if (body !== undefined && !token.current) await session();
  const send = () =>
    boundedJson<T>(
      path,
      {
        method: body === undefined ? 'GET' : 'POST',
        headers:
          body === undefined
            ? {}
            : {
                'content-type': 'application/json',
                'x-lerobot-token': token.current,
              },
        body: body === undefined ? undefined : JSON.stringify(body),
      },
      fetcher,
      timeoutMs,
    );
  try {
    return await send();
  } catch (error) {
    // Only an explicit auth rejection may be retried. Never retry an uncertain
    // network timeout/500, which could otherwise start the same motion twice.
    if (
      body === undefined ||
      !(error instanceof ControllerRequestError) ||
      error.status !== 403
    )
      throw error;
    token.current = '';
    await session();
    try {
      return await send();
    } catch (retryError) {
      if (
        retryError instanceof ControllerRequestError &&
        retryError.status === 403
      )
        token.current = '';
      throw retryError;
    }
  }
}

type BasicJob = { id: string; status: string };
type Read = <T>(path: string) => Promise<T>;
export async function readControllerPoll<
  J extends BasicJob,
  S extends { job: J | null },
>(
  read: Read,
  trackedId: string,
  isCurrent: () => boolean,
): Promise<
  | { kind: 'stale' }
  | {
      kind: 'snapshot';
      status: S;
      current: J | null;
      activeId: string;
      missingId: string | null;
    }
> {
  try {
    const status = await read<S>('/api/status');
    if (!isCurrent()) return { kind: 'stale' };
    let current = status.job;
    let activeId = trackedId;
    if (current?.status === 'running' && current.id !== activeId)
      activeId = current.id;
    else if (activeId && current?.id !== activeId) {
      try {
        current = await read<J>(`/api/jobs/${encodeURIComponent(activeId)}`);
      } catch (error) {
        if (!isCurrent()) return { kind: 'stale' };
        if (error instanceof ControllerRequestError && error.status === 404)
          return {
            kind: 'snapshot',
            status,
            current: null,
            activeId: '',
            missingId: activeId,
          };
        throw error; // Keep the job/partial result on transient connection errors.
      }
    }
    if (!isCurrent()) return { kind: 'stale' };
    return { kind: 'snapshot', status, current, activeId, missingId: null };
  } catch (error) {
    if (!isCurrent()) return { kind: 'stale' };
    throw error;
  }
}

export function progressPresentation(
  sending: boolean,
  job: BasicJob | null,
  serverJob: BasicJob | null | undefined,
  connectionIssue: boolean,
  stopAcknowledgedId: string,
) {
  const working =
    sending || job?.status === 'running' || serverJob?.status === 'running';
  const stopped = !!stopAcknowledgedId && job?.id === stopAcknowledgedId;
  return {
    busy: !!working || connectionIssue,
    showProgress: !!working && !connectionIssue && !stopped,
    stopped,
  };
}
